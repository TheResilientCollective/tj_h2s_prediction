"""
HYSPLIT runner — execute CONTROL bundles produced by hysplit_controls.py.

This module is the execution-side complement to `hysplit_controls.generate_hysplit_bundle`.
It unzips a bundle, invokes the appropriate HYSPLIT executable (`hyts_std` for
trajectories, `hycs_std` for dispersion) per CONTROL file, captures MESSAGE-file
diagnostics, and collects output (tdump/cdump) files.

Designed to be invoked from a Dagster op inside the dedicated HYSPLIT worker
container. All paths are sourced from the environment so the same module works
in containers and locally:

    HYSPLIT_PATH         e.g. /opt/hysplit/exec
    HYSPLIT_METEO_DIR    e.g. /data/hysplit/meteo
    HYSPLIT_WORKING_DIR  e.g. /data/hysplit/work
    HYSPLIT_OUTPUT_DIR   e.g. /data/hysplit/output

Ported from GeoDemic/backend/app/services/hysplit_service.py (subprocess
invocation + MESSAGE diagnostics), stripped of mock-data and weather-service
fallbacks. Bundle generation is intentionally NOT duplicated — this module
consumes zip bytes produced by the existing generator.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


# Mode → executable map. Mirrors hysplit_controls._write_run_script exe_map.
MODE_EXECUTABLE = {
    "backward_traj": "hyts_std",
    "backward_disp": "hycs_std",
    "forward_disp":  "hycs_std",
}


@dataclass
class RunResult:
    """Result of executing a single CONTROL file."""
    control_name: str
    returncode: int
    output_files: list[Path]
    message: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0 and len(self.output_files) > 0


@dataclass
class RunOutputs:
    """Aggregated result of running a full bundle."""
    mode: str
    results: list[RunResult] = field(default_factory=list)
    output_paths: list[Path] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def messages(self) -> list[str]:
        return [r.message for r in self.results if r.message]


class HysplitRunner:
    """
    Execute HYSPLIT CONTROL files against a pre-installed HYSPLIT binary.

    Configuration is read from env on construction; each call to
    `run_bundle_zip` creates a fresh per-run working directory so multiple
    concurrent runs do not clobber each other's CONTROL/MESSAGE files.
    """

    def __init__(
        self,
        hysplit_path: Optional[str] = None,
        meteo_dir: Optional[str] = None,
        working_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout_seconds: int = 1800,
    ):
        self.hysplit_path = Path(
            hysplit_path or os.environ.get("HYSPLIT_PATH", "/opt/hysplit/exec")
        )
        self.meteo_dir = Path(
            meteo_dir or os.environ.get("HYSPLIT_METEO_DIR", "/data/hysplit/meteo")
        )
        self.working_dir = Path(
            working_dir or os.environ.get("HYSPLIT_WORKING_DIR", "/data/hysplit/work")
        )
        self.output_dir = Path(
            output_dir or os.environ.get("HYSPLIT_OUTPUT_DIR", "/data/hysplit/output")
        )
        self.timeout_seconds = timeout_seconds

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_bundle_zip(self, zip_bytes: bytes, mode: str) -> RunOutputs:
        """
        Unzip a HYSPLIT bundle and execute every CONTROL file it contains.

        The bundle layout is produced by
        `h2s.dispersion.hysplit_controls.generate_hysplit_bundle`:

            <mode>/
                SETUP.CFG
                CONTROL.<tag1>
                CONTROL.<tag2>
                ...
                run_hysplit_<mode>.sh   (ignored here — we drive subprocess directly)
                output/                 (empty placeholder)

        Outputs are moved into a per-run subdirectory under `output_dir` and
        their paths are returned for upload by the caller.
        """
        if mode not in MODE_EXECUTABLE:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected one of {sorted(MODE_EXECUTABLE)}."
            )
        executable = self.hysplit_path / MODE_EXECUTABLE[mode]
        if not executable.exists():
            raise FileNotFoundError(
                f"HYSPLIT executable not found: {executable}. "
                f"Check HYSPLIT_PATH (currently {self.hysplit_path})."
            )

        # Per-run working + output subdirs — isolate concurrent calls.
        run_id = os.urandom(4).hex()
        run_work = self.working_dir / f"run_{run_id}"
        run_out = self.output_dir / f"run_{run_id}"
        run_work.mkdir(parents=True, exist_ok=True)
        run_out.mkdir(parents=True, exist_ok=True)

        try:
            extract_dir = run_work / "bundle"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(extract_dir)

            # Bundles are generated with a top-level <mode>/ subdirectory.
            # Fall back to extract_dir itself if the layout is flat.
            candidates = [extract_dir / mode, extract_dir]
            bundle_dir = next((c for c in candidates if c.is_dir()), None)
            if bundle_dir is None:
                raise RuntimeError(f"Bundle extracted but no directory found at {candidates!r}")

            setup_cfg = bundle_dir / "SETUP.CFG"
            control_files = sorted(bundle_dir.glob("CONTROL.*"))
            if not control_files:
                raise RuntimeError(
                    f"No CONTROL.* files found in bundle at {bundle_dir}"
                )

            logger.info(
                "Running HYSPLIT bundle: mode=%s, executable=%s, n_controls=%d",
                mode, executable, len(control_files),
            )

            outputs = RunOutputs(mode=mode)
            for ctrl_path in control_files:
                result = self.run_control(
                    ctrl_path,
                    executable=executable,
                    setup_cfg=setup_cfg if setup_cfg.exists() else None,
                    run_work=run_work,
                    run_out=run_out,
                )
                outputs.results.append(result)
                outputs.output_paths.extend(result.output_files)

            return outputs

        finally:
            # Clean working dir but KEEP output dir — caller reads it.
            shutil.rmtree(run_work, ignore_errors=True)

    def run_control(
        self,
        control_path: Path,
        executable: Path,
        setup_cfg: Optional[Path],
        run_work: Path,
        run_out: Path,
    ) -> RunResult:
        """
        Execute one CONTROL file.

        HYSPLIT's convention is rigid: it reads `CONTROL` (no suffix) from
        the current working directory and writes `MESSAGE` alongside it.
        We copy the per-run CONTROL.<tag> into a dedicated subdirectory as
        `CONTROL`, copy SETUP.CFG next to it if present, then invoke the
        executable with that subdirectory as cwd.
        """
        ctrl_name = control_path.name
        ctrl_subdir = run_work / ctrl_name
        ctrl_subdir.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(control_path, ctrl_subdir / "CONTROL")
        if setup_cfg is not None:
            shutil.copyfile(setup_cfg, ctrl_subdir / "SETUP.CFG")

        logger.info("Executing HYSPLIT for %s in %s", ctrl_name, ctrl_subdir)
        try:
            proc = subprocess.run(
                [str(executable)],
                cwd=str(ctrl_subdir),
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            returncode = proc.returncode
            stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        except subprocess.TimeoutExpired as exc:
            logger.error("HYSPLIT timed out after %ds for %s", self.timeout_seconds, ctrl_name)
            returncode = -1
            stderr = f"TimeoutExpired after {self.timeout_seconds}s: {exc}"

        # MESSAGE file always holds HYSPLIT diagnostics, even on failure.
        message_path = ctrl_subdir / "MESSAGE"
        message = ""
        if message_path.exists():
            try:
                message = message_path.read_text(errors="replace")
            except Exception as exc:
                logger.warning("Could not read MESSAGE for %s: %s", ctrl_name, exc)

        if returncode != 0:
            logger.warning(
                "HYSPLIT exited %d for %s | stderr=%r | MESSAGE=%s",
                returncode, ctrl_name, stderr[:300], (message or "(none)")[:500],
            )

        # Collect output files: tdump/cdump + anything named in the CONTROL,
        # while excluding the rigid HYSPLIT control/diagnostic files.
        excluded = {"CONTROL", "SETUP.CFG", "MESSAGE", "CONC.CFG", "WARNING", "STARTUP"}
        output_files: list[Path] = []
        tag = ctrl_name.split(".", 1)[1] if "." in ctrl_name else ctrl_name
        dest_dir = run_out / tag
        dest_dir.mkdir(parents=True, exist_ok=True)

        for item in ctrl_subdir.iterdir():
            if item.is_file() and item.name not in excluded:
                dest = dest_dir / item.name
                try:
                    shutil.move(str(item), str(dest))
                    output_files.append(dest)
                except Exception as exc:
                    logger.error("Could not move %s → %s: %s", item, dest, exc)

        # Also preserve the MESSAGE file for auditing.
        if message_path.exists():
            msg_dest = dest_dir / "MESSAGE"
            try:
                shutil.copyfile(message_path, msg_dest)
            except Exception:
                pass

        return RunResult(
            control_name=ctrl_name,
            returncode=returncode,
            output_files=output_files,
            message=message,
            stderr=stderr,
        )
