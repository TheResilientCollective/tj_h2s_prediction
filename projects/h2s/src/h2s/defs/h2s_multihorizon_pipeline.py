"""Multi-Horizon H2S Forecast Pipeline.

Loads 36 pre-trained MH models from S3, runs horizon-aware 72h forecasts,
generates dashboard visualizations, and exports results.
"""

import io
import json
import os
import pickle
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import FORECAST_DATA_PATH, MH_MODELS_S3_BASE, MH_OUTPUT_PATH, LATEST_BASEPATH, OBS_DATA_PATH
from h2s.training.multihorizon_trainer import (
    BASE_FEATURES,
    FLOW_COL,
    HORIZONS,
    HORIZON_BOUNDS,
    HORIZON_NAMES,
    STATIONS,
    TASKS,
    build_forecast_features,
    classify_risk,
    find_aligned_source,
    get_obs_state,
)

_KEY = lambda name: dg.AssetKey(["h2s", name])
ENV_LABEL = os.environ.get("ENV_LABEL", "add_ENV_LABEL").upper()

# ==============================================================================
# Asset 1: Load multi-horizon models from S3
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_forecast",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Load 36 multi-horizon models + horizon_features.json from S3",
)
def mh_model_artifacts(context: dg.AssetExecutionContext) -> dict:
    """Load all MH models from S3.

    Returns dict with:
      - 'models': {horizon: {station_key: {task: model}}}
      - 'horizon_features': {horizon: {station_name: [feature_cols]}}
    """
    # Import here to ensure EnsembleRegressor/EnsembleClassifier are in scope for unpickling
    from h2s.training.multihorizon_trainer import EnsembleRegressor, EnsembleClassifier  # noqa: F401

    s3 = context.resources.s3
    bucket = s3.S3_BUCKET

    models = {}
    loaded_count = 0

    for hz_name in HORIZON_NAMES:
        models[hz_name] = {}
        for site_name, sinfo in STATIONS.items():
            skey = sinfo['key']
            models[hz_name][skey] = {}
            for task in TASKS:
                s3_path = f"{MH_MODELS_S3_BASE}/{hz_name}/{skey}/{task}.pkl"
                try:
                    model_bytes = s3.getFile(s3_path, bucket=bucket)
                    model = pickle.loads(model_bytes)
                    models[hz_name][skey][task] = model
                    loaded_count += 1
                except Exception as e:
                    context.log.warning(f"Could not load {s3_path}: {e}")

    # Load horizon features
    horizon_features = {}
    for site_name, sinfo in STATIONS.items():
        skey = sinfo['key']
        feat_path = f"{MH_MODELS_S3_BASE}/{skey}/horizon_features.json"
        try:
            feat_bytes = s3.getFile(feat_path, bucket=bucket)
            station_features = json.loads(feat_bytes.decode('utf-8'))
            for hz_name, cols in station_features.items():
                if hz_name not in horizon_features:
                    horizon_features[hz_name] = {}
                horizon_features[hz_name][site_name] = cols
        except Exception as e:
            context.log.warning(f"Could not load horizon features for {skey}: {e}")

    context.log.info(f"Loaded {loaded_count} MH models from S3")
    context.add_output_metadata({
        "models_loaded": loaded_count,
        "horizons": list(models.keys()),
    })

    return {"models": models, "horizon_features": horizon_features}


# ==============================================================================
# Asset 2: Extract observation state for lag seeding
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_forecast",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Load recent observation data from S3 to seed H2S lag features in forecasts",
    config_schema={
        "s3_bucket": dg.Field(
            str,
            default_value="resilentpublic",
            description="S3 bucket for observation data",
        ),
    },
)
def mh_observation_state(context: dg.AssetExecutionContext) -> dict:
    """Load observation parquet and extract per-station state for lag seeding."""
    s3 = context.resources.s3
    bucket = context.op_config["s3_bucket"]
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

    parquet_url = s3.get_presigned_url(path=s3_path, bucket=bucket)
    obs_df = pd.read_parquet(parquet_url)
    obs_df['time'] = pd.to_datetime(obs_df['time'], utc=True)
    obs_df = obs_df[(obs_df['h2s_measured'] == True) & (obs_df['H2S'] <= 500)].copy()
    obs_df['H2S'] = obs_df['H2S'].clip(lower=0)

    context.log.info(f"Loaded observations: {len(obs_df)} rows through {obs_df['time'].max()}")

    states = {}
    for site_name in STATIONS:
        states[site_name] = get_obs_state(obs_df, site_name)
        n_obs = len(states[site_name]['h2s_series'])
        context.log.info(f"  {site_name}: {n_obs} observations, last H2S={states[site_name]['last_h2s']:.1f}")

    context.add_output_metadata({
        "obs_rows": len(obs_df),
        "obs_through": str(obs_df['time'].max()),
        "stations": list(states.keys()),
    })
    return states


# ==============================================================================
# Asset 3: Run multi-horizon forecasts
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_forecast",
    required_resource_keys={"s3"},
    kinds={"python", "ml"},
    description="Run 72h multi-horizon H2S forecast for all stations",
    ins={
        "mh_model_artifacts": dg.AssetIn(key=_KEY("mh_model_artifacts")),
        "mh_observation_state": dg.AssetIn(key=_KEY("mh_observation_state")),
    },
    config_schema={
        "s3_bucket": dg.Field(str, default_value="resilentpublic"),
    },
)
def mh_forecasts(
    context: dg.AssetExecutionContext,
    mh_model_artifacts: dict,
    mh_observation_state: dict,
) -> pd.DataFrame:
    """Run multi-horizon forecast using pre-featurized forecast data from S3.

    Loads model_forecast.parquet directly from S3, assigns each forecast hour
    to its horizon bucket, builds horizon-specific features, and runs predictions.
    """
    s3 = context.resources.s3
    bucket = context.op_config["s3_bucket"]

    # Load forecast parquet from S3
    context.log.info(f"Loading forecast data from S3: {FORECAST_DATA_PATH}")
    try:
        fc_url = s3.get_presigned_url(path=FORECAST_DATA_PATH, bucket=bucket)
        forecast_df = pd.read_parquet(fc_url)
        context.log.info(f"✓ Loaded {len(forecast_df)} rows from S3")
    except Exception as e:
        raise RuntimeError(f"Failed to load forecast data from S3 path '{FORECAST_DATA_PATH}': {e}")

    forecast_df['time'] = pd.to_datetime(forecast_df['time'], utc=True)
    fc_start = forecast_df['time'].min()

    models = mh_model_artifacts['models']
    horizon_features = mh_model_artifacts.get('horizon_features', {})

    all_results = []

    for site_name, sinfo in STATIONS.items():
        skey = sinfo['key']
        obs_state = mh_observation_state.get(site_name)
        if obs_state is None:
            context.log.warning(f"No observation state for {site_name}, skipping")
            continue

        sfc = forecast_df[forecast_df['site_name'] == site_name].sort_values('time').reset_index(drop=True)
        if len(sfc) == 0:
            context.log.warning(f"No forecast data for {site_name}")
            continue

        sfc['hours_ahead'] = (sfc['time'] - fc_start).dt.total_seconds() / 3600

        for hz_name, h_start, h_end in HORIZON_BOUNDS:
            hz_cfg = HORIZONS[hz_name]
            mask = (sfc['hours_ahead'] >= h_start) & (sfc['hours_ahead'] < h_end)
            hz_slice = sfc[mask].copy()
            if len(hz_slice) == 0:
                continue

            # Build horizon-specific features
            hz_feat, feature_cols = build_forecast_features(hz_slice, obs_state, hz_name, hz_cfg)

            # Use stored feature list if available (ensures column order matches training)
            stored_cols = horizon_features.get(hz_name, {}).get(site_name)
            if stored_cols:
                feature_cols = stored_cols

            # Check models exist
            if skey not in models.get(hz_name, {}):
                context.log.warning(f"No models for {hz_name}/{skey}")
                continue
            hz_models = models[hz_name][skey]
            if not all(t in hz_models for t in TASKS):
                context.log.warning(f"Incomplete models for {hz_name}/{skey}: {list(hz_models.keys())}")
                continue

            # Ensure all feature columns exist, fill missing with 0
            for col in feature_cols:
                if col not in hz_feat.columns:
                    hz_feat[col] = 0.0

            X = hz_feat[feature_cols].fillna(0).values
            h2s_pred = np.clip(hz_models['regression'].predict(X), 0, None)
            prob_5 = hz_models['clf_5ppb'].predict_proba(X)[:, 1]
            prob_10 = hz_models['clf_10ppb'].predict_proba(X)[:, 1]

            for i in range(len(hz_slice)):
                wd = float(hz_slice['wind_direction_10m'].iloc[i])
                is_n = bool(hz_slice['is_night'].iloc[i])

                all_results.append({
                    'time': hz_slice['time'].iloc[i],
                    'station': site_name,
                    'horizon': hz_name,
                    'hours_ahead': round(float(hz_slice['hours_ahead'].iloc[i]), 1),
                    'h2s_pred': round(float(h2s_pred[i]), 1),
                    'prob_5': round(float(prob_5[i]) * 100, 1),
                    'prob_10': round(float(prob_10[i]) * 100, 1),
                    'risk': classify_risk(prob_5[i], prob_10[i], h2s_pred[i]),
                    'wind_speed': round(float(hz_slice['wind_speed_10m'].iloc[i]), 1),
                    'wind_dir': round(wd),
                    'temp': round(float(hz_slice['temperature_2m'].iloc[i]), 1),
                    'tide': round(float(hz_slice['tide_height'].iloc[i]), 2),
                    'flow': round(float(hz_slice[FLOW_COL].iloc[i]), 2) if FLOW_COL in hz_slice.columns else 0.0,
                    'sbiwtp': round(float(hz_slice['sbiwtp_flow_mgd'].iloc[i]), 1) if 'sbiwtp_flow_mgd' in hz_slice.columns else 0.0,
                    'is_night': int(is_n),
                    'aligned_source': find_aligned_source(site_name, wd, is_n),
                })

    results = pd.DataFrame(all_results)
    context.log.info(f"MH forecast complete: {len(results)} rows across {results['station'].nunique()} stations")

    if len(results) > 0:
        for site in results['station'].unique():
            sf = results[results['station'] == site]
            rc = sf['risk'].value_counts().to_dict()
            context.log.info(
                f"  {site}: max={sf['h2s_pred'].max():.0f}ppb "
                f"G:{rc.get('GREEN',0)} YL:{rc.get('YELLOW_LOW',0)} YH:{rc.get('YELLOW_HIGH',0)} O:{rc.get('ORANGE',0)}"
            )

    context.add_output_metadata({
        "forecast_rows": len(results),
        "stations": list(results['station'].unique()) if len(results) > 0 else [],
        "forecast_start": str(results['time'].min()) if len(results) > 0 else "N/A",
        "forecast_end": str(results['time'].max()) if len(results) > 0 else "N/A",
    })
    return results


# ==============================================================================
# Asset 4: Dashboard visualization
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_forecast",
    required_resource_keys={"s3"},
    kinds={"python", "matplotlib"},
    description="Multi-horizon forecast dashboard (dark-themed PNG)",
    ins={
        "mh_forecasts": dg.AssetIn(key=_KEY("mh_forecasts")),
        "mh_observation_state": dg.AssetIn(key=_KEY("mh_observation_state")),
    },
)
def mh_dashboard_viz(
    context: dg.AssetExecutionContext,
    mh_forecasts: pd.DataFrame,
    mh_observation_state: dict,
) -> None:
    """Generate and upload multi-horizon dashboard to S3."""
    if len(mh_forecasts) == 0:
        context.log.warning("No forecast data, skipping dashboard")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    results = mh_forecasts
    hz_colors = {'0_6h': '#4fc3f7', '6_24h': '#66bb6a', '24_48h': '#ffa726', '48_72h': '#ef5350'}
    station_colors = {'SAN YSIDRO': '#e74c3c', 'NESTOR - BES': '#2ecc71', 'IB CIVIC CTR': '#3498db'}
    risk_colors = {'GREEN': '#27ae60', 'YELLOW_LOW': '#f39c12', 'YELLOW_HIGH': '#e67e22', 'ORANGE': '#e74c3c'}

    station_list = [s for s in STATIONS if s in results['station'].unique()]

    fig = plt.figure(figsize=(24, 20))
    fig.set_facecolor('#0f0f1a')
    gs = fig.add_gridspec(4, max(len(station_list), 1), hspace=0.35, wspace=0.25)
    _pacific = ZoneInfo("America/Los_Angeles")
    _t_min_pt = results["time"].min().astimezone(_pacific)
    _t_max_pt = results["time"].max().astimezone(_pacific)
    fig.suptitle(
        f'H\u2082S Multi-Horizon Forecast\n'
        f'{_t_min_pt.strftime("%Y-%m-%d %H:%M")} \u2014 '
        f'{_t_max_pt.strftime("%Y-%m-%d %H:%M")} PT',
        fontsize=16, fontweight='bold', color='white', y=0.99
    )

    for idx, site in enumerate(station_list):
        sf = results[results['station'] == site].sort_values('time')
        obs_state = mh_observation_state.get(site, {})

        # Row 1: H2S with horizon coloring
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#1a1a2e')

        for hz_name, _, _ in HORIZON_BOUNDS:
            hz = sf[sf['horizon'] == hz_name]
            if len(hz) > 0:
                ax.plot(hz['time'], hz['h2s_pred'], color=hz_colors[hz_name], linewidth=2, label=hz_name)
                ax.fill_between(hz['time'], 0, hz['h2s_pred'], alpha=0.1, color=hz_colors[hz_name])

        ax.axhline(5, color='#f39c12', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.axhline(30, color='#e74c3c', linewidth=0.5, linestyle='--', alpha=0.4)

        rc = sf[sf['hours_ahead'] <= 48]['risk'].value_counts().to_dict()
        ax.set_title(
            f'{site}\n48h: G:{rc.get("GREEN",0)} YL:{rc.get("YELLOW_LOW",0)} YH:{rc.get("YELLOW_HIGH",0)} O:{rc.get("ORANGE",0)}',
            fontsize=10, fontweight='bold', color='white'
        )
        ax.set_ylabel('H\u2082S (ppb)', color='white')
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
        ax.grid(True, alpha=0.15, color='white')
        for s in ax.spines.values():
            s.set_color('#333')

        # Row 2: Exceedance probability
        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor('#1a1a2e')
        ax.fill_between(sf['time'], 0, sf['prob_5'], alpha=0.3, color='#f39c12', label='P(>5)')
        ax.plot(sf['time'], sf['prob_5'], color='#f39c12', linewidth=1.5)
        ax.fill_between(sf['time'], 0, sf['prob_10'], alpha=0.4, color='#e74c3c', label='P(>10)')
        ax.plot(sf['time'], sf['prob_10'], color='#e74c3c', linewidth=1.5)
        ax.axhline(50, color='white', linewidth=0.5, linestyle=':', alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)', color='white')
        ax.set_title('Exceedance Probability', fontsize=10, fontweight='bold', color='white')
        ax.legend(fontsize=7)
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
        ax.grid(True, alpha=0.15, color='white')
        for s in ax.spines.values():
            s.set_color('#333')

        # Row 3: Risk timeline
        ax = fig.add_subplot(gs[2, idx])
        ax.set_facecolor('#1a1a2e')
        for _, row in sf.iterrows():
            ax.bar(row['time'], 1, width=pd.Timedelta(hours=1),
                   color=risk_colors.get(row['risk'], '#333'), alpha=0.8)
        for hz, hs, he in HORIZON_BOUNDS:
            t = sf['time'].min() + pd.Timedelta(hours=hs)
            if t <= sf['time'].max():
                ax.axvline(t, color='white', linewidth=0.8, linestyle=':', alpha=0.5)
                ax.text(t, 0.5, hz, fontsize=7, color='white', alpha=0.6, rotation=90, va='center')
        ax.set_yticks([])
        ax.set_ylabel('Risk', color='white')
        ax.set_title('Hourly Risk Tier', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
        for s in ax.spines.values():
            s.set_color('#333')

        # Row 4: Weather
        ax = fig.add_subplot(gs[3, idx])
        ax.set_facecolor('#1a1a2e')
        ax.plot(sf['time'], sf['temp'], color='#feca57', linewidth=1.5, label='Temp')
        ax.set_ylabel('Temp (\u00b0C)', color='#feca57')
        ax2 = ax.twinx()
        ax2.plot(sf['time'], sf['wind_speed'], color='#48dbfb', linewidth=1, alpha=0.7, label='Wind')
        ax2.set_ylabel('Wind (m/s)', color='#48dbfb')
        ax2.tick_params(colors='#48dbfb', labelsize=7)
        ax.set_title('Weather', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
        ax.grid(True, alpha=0.15, color='white')
        for s in ax.spines.values():
            s.set_color('#333')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor='#0f0f1a', format='png')
    plt.close()
    buf.seek(0)

    # Upload to S3
    s3 = context.resources.s3
    now = datetime.now(timezone.utc)
    date_str = now.strftime('%Y-%m-%d')

    timestamped_path = f"{MH_OUTPUT_PATH}/visualizations/{date_str}/mh_dashboard.png"
    latest_path = f"{LATEST_BASEPATH}/multihorizon/mh_dashboard.png"

    s3.putFile(buf.getvalue(), timestamped_path, bucket=s3.S3_BUCKET, content_type='image/png')
    s3.putFile(buf.getvalue(), latest_path, bucket=s3.S3_BUCKET, content_type='image/png')
    context.log.info(f"Dashboard uploaded to {timestamped_path} and {latest_path}")

    context.add_output_metadata({
        "s3_path": timestamped_path,
        "latest_path": latest_path,
    })


# ==============================================================================
# Asset 5: Export forecast results
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_forecast",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Export MH forecast CSV + JSON summary to S3",
    ins={"mh_forecasts": dg.AssetIn(key=_KEY("mh_forecasts"))},
)
def mh_summary_export(
    context: dg.AssetExecutionContext,
    mh_forecasts: pd.DataFrame,
) -> dict:
    """Export forecast results as CSV and JSON summary to S3."""
    if len(mh_forecasts) == 0:
        context.log.warning("No forecast data to export")
        return {"status": "empty"}

    s3 = context.resources.s3
    now = datetime.now(timezone.utc)
    date_str = now.strftime('%Y-%m-%d')
    results = mh_forecasts

    # CSV export
    csv_buf = io.BytesIO()
    results.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue()

    timestamped_csv = f"{MH_OUTPUT_PATH}/{date_str}/forecast_mh.csv"
    latest_csv = f"{LATEST_BASEPATH}/multihorizon/forecast_mh.csv"
    s3.putFile(csv_bytes, timestamped_csv, bucket=s3.S3_BUCKET, content_type='text/csv')
    s3.putFile(csv_bytes, latest_csv, bucket=s3.S3_BUCKET, content_type='text/csv')

    # JSON summary
    summary = {
        'generated_at': now.isoformat(),
        'forecast_start': str(results['time'].min()),
        'forecast_end': str(results['time'].max()),
        'stations': {},
    }
    for site in results['station'].unique():
        sf = results[results['station'] == site]
        by_hz = {}
        for hz_name, _, _ in HORIZON_BOUNDS:
            hz = sf[sf['horizon'] == hz_name]
            if len(hz) == 0:
                continue
            rc = hz['risk'].value_counts().to_dict()
            by_hz[hz_name] = {
                'max_h2s': round(float(hz['h2s_pred'].max()), 1),
                'max_prob_5': round(float(hz['prob_5'].max()), 1),
                'max_prob_10': round(float(hz['prob_10'].max()), 1),
                'hours_orange': int(rc.get('ORANGE', 0)),
                'hours_yellow_high': int(rc.get('YELLOW_HIGH', 0)),
                'hours_yellow_low': int(rc.get('YELLOW_LOW', 0)),
                'hours_green': int(rc.get('GREEN', 0)),
            }
        summary['stations'][site] = by_hz

    summary_bytes = json.dumps(summary, indent=2, default=str).encode('utf-8')
    timestamped_json = f"{MH_OUTPUT_PATH}/{date_str}/forecast_summary_mh.json"
    latest_json = f"{LATEST_BASEPATH}/multihorizon/forecast_summary_mh.json"
    s3.putFile(summary_bytes, timestamped_json, bucket=s3.S3_BUCKET, content_type='application/json')
    s3.putFile(summary_bytes, latest_json, bucket=s3.S3_BUCKET, content_type='application/json')

    context.log.info(f"Exported CSV ({len(csv_bytes):,} bytes) and JSON summary to S3")

    context.add_output_metadata({
        "csv_path": timestamped_csv,
        "json_path": timestamped_json,
        "forecast_rows": len(results),
        "stations": list(results['station'].unique()),
    })
    return summary


# ==============================================================================
# Asset 6: Slack alerts
# ==============================================================================

_RISK_EMOJI = {'ORANGE': '🟠', 'YELLOW_HIGH': '🟡', 'YELLOW_LOW': '🔆', 'GREEN': '🟢'}
_RISK_ORDER = ['ORANGE', 'YELLOW_HIGH', 'YELLOW_LOW', 'GREEN']
_HZ_LABELS = {'0_6h': '0-6h', '6_24h': '6-24h', '24_48h': '24-48h', '48_72h': '48-72h'}


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_mh_forecast",
    required_resource_keys={"slack"},
    kinds={"slack"},
    description="Send multi-horizon H2S elevated-risk alerts to Slack",
    ins={"mh_forecasts": dg.AssetIn(key=_KEY("mh_forecasts"))},
)
def mh_slack_alerts(
    context: dg.AssetExecutionContext,
    mh_forecasts: pd.DataFrame,
) -> None:
    """Post MH forecast alert to Slack when ORANGE or YELLOW_HIGH hours are predicted.

    Skips silently if all stations show GREEN/YELLOW_LOW across all horizons.
    Groups results by station with a per-horizon breakdown of elevated hours.
    """
    if len(mh_forecasts) == 0:
        context.log.info("No MH forecast data — skipping Slack alert")
        return

    elevated = mh_forecasts[mh_forecasts['risk'].isin(['ORANGE', 'YELLOW_HIGH'])]
    if elevated.empty:
        context.log.info("No elevated risk predicted — skipping Slack alert")
        context.add_output_metadata({"status": "skipped", "reason": "all green/yellow_low"})
        return

    # Count unique hours per horizon where ANY station shows elevated risk
    hz_summary_parts = []
    total_orange = 0
    total_yh = 0
    for hz_key, hz_label in _HZ_LABELS.items():
        hz_df = mh_forecasts[mh_forecasts['horizon'] == hz_key]
        if hz_df.empty:
            hz_summary_parts.append(f"{hz_label}: 🟢 0h")
            continue
        hz_hourly = hz_df.groupby('time')['risk'].agg(
            lambda x: 'ORANGE' if 'ORANGE' in x.values
            else ('YELLOW_HIGH' if 'YELLOW_HIGH' in x.values else x.iloc[0])
        )
        hz_orange = int((hz_hourly == 'ORANGE').sum())
        hz_yh = int((hz_hourly == 'YELLOW_HIGH').sum())
        total_orange += hz_orange
        total_yh += hz_yh
        worst = 'ORANGE' if hz_orange > 0 else ('YELLOW_HIGH' if hz_yh > 0 else 'GREEN')
        elevated = hz_orange + hz_yh
        hz_summary_parts.append(f"{hz_label}: {_RISK_EMOJI[worst]} {elevated}h")

    blocks: list = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{ENV_LABEL} ⚠ Multi-Horizon H2S Alert — Elevated Levels Forecast"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*🟠 Orange:* {total_orange}h  |  *🟡 Yellow-High:* {total_yh}h\n"
                    + " | ".join(hz_summary_parts)
                ),
            },
        },
        {"type": "divider"},
    ]

    for site_name in STATIONS:
        sf = mh_forecasts[mh_forecasts['station'] == site_name]
        if sf.empty:
            continue

        max_pred = float(sf['h2s_pred'].max())
        worst_risk = next(
            (r for r in _RISK_ORDER if (sf['risk'] == r).any()), 'GREEN'
        )

        hz_parts = []
        for hz_key, hz_label in _HZ_LABELS.items():
            hz = sf[sf['horizon'] == hz_key]
            if hz.empty:
                hz_parts.append(f"{hz_label}: {_RISK_EMOJI['GREEN']} 0h")
                continue
            worst_hz = next((r for r in _RISK_ORDER if (hz['risk'] == r).any()), 'GREEN')
            elevated_h = int((hz['risk'].isin(['ORANGE', 'YELLOW_HIGH'])).sum())
            hz_parts.append(f"{hz_label}: {_RISK_EMOJI[worst_hz]} {elevated_h}h")

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{_RISK_EMOJI[worst_risk]} *{site_name}* — peak {max_pred:.0f} ppb\n"
                    + " | ".join(hz_parts)
                ),
            },
        })

    pacific = ZoneInfo("America/Los_Angeles")
    t_min = mh_forecasts['time'].min().astimezone(pacific).strftime("%-I %p %-m/%-d")
    t_max = mh_forecasts['time'].max().astimezone(pacific).strftime("%-I %p %-m/%-d")
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"Forecast window: {t_min} → {t_max} PT"}],
    })

    slack = context.resources.slack
    slack.get_client().chat_postMessage(
        channel=slack.channel,
        text=f"MH H2S Alert: {total_orange} orange, {total_yh} yellow-high unique hours",
        blocks=blocks,
    )

    context.log.info(f"MH Slack alert sent: {total_orange} orange, {total_yh} yellow-high hours")
    context.add_output_metadata({
        "orange_hours": total_orange,
        "yellow_high_hours": total_yh,
        "stations_alerted": int(mh_forecasts[mh_forecasts['risk'].isin(['ORANGE', 'YELLOW_HIGH'])]['station'].nunique()),
    })


# ==============================================================================
# Job definition
# ==============================================================================

mh_forecast_job = dg.define_asset_job(
    name="mh_forecast_job",
    description="Run multi-horizon H2S forecast: load models, predict 72h, generate dashboard + exports",
    selection=dg.AssetSelection.assets(
        mh_model_artifacts,
        mh_observation_state,
        mh_forecasts,
        mh_dashboard_viz,
        mh_summary_export,
        mh_slack_alerts,
    ),
    tags={"environment": "production", "pipeline": "h2s_mh_forecast"},
)
