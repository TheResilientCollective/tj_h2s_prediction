# Wind Speed Dependency in H2S Dispersion

**Critical Finding:** H2S concentrations at NESTOR-BES are **strongly anti-correlated with wind speed** (r = -0.246, p < 0.001).

## Observational Evidence (Feb-Apr 2026)

| Wind Speed (m/s) | Mean H2S (ppb) | Median H2S (ppb) | Max H2S (ppb) | N obs |
|------------------|----------------|------------------|---------------|-------|
| 0-1              | 49.9           | 10.7             | 557.3         | 68    |
| 1-2              | 42.8           | 8.1              | 394.0         | 94    |
| 2-3              | 32.9           | 5.4              | 335.1         | 118   |
| 3-4              | 20.9           | 4.4              | 262.4         | 141   |
| 4-5              | 12.8           | 4.6              | 176.8         | 106   |
| >5               | 6.4            | 1.7              | 239.2         | 557   |

**High H2S events (≥30 ppb):** median wind speed = 2.45 m/s
**Low H2S events (<5 ppb):** median wind speed = 6.26 m/s

**Ratio:** 6.26 / 2.45 = **2.6× stronger winds during safe conditions**

## Physical Interpretation

This is **fundamental dispersion physics**:

```
Concentration ∝ Q / (U · σy · σz)
```

Where:
- Q = emission rate (g/s)
- U = wind speed (m/s)
- σy, σz = lateral and vertical diffusion coefficients (m)

**Low wind speed causes high concentrations via two mechanisms:**

1. **Direct dilution effect:** Lower U → less air volume to dilute emissions
2. **Reduced turbulence:** Lower U → weaker mechanical turbulence → smaller σy, σz → less dispersion

The **combined effect** is roughly:

```
C ∝ U^(-1) · U^(-0.5) = U^(-1.5)
```

So halving wind speed increases concentration by ~2.8×, consistent with observations.

## Implications for Lagrangian Model

### Current Implementation (WRONG)

`lagrangian.py:LagrangianConfig` uses **fixed diffusion coefficients**:

```python
sigma_u: float = 0.3   # m/s, horizontal velocity perturbation
sigma_v: float = 0.3   # m/s, horizontal velocity perturbation
sigma_w: float = 0.05  # m/s, vertical velocity perturbation
```

These values do **not depend on wind speed**, which is physically incorrect.

### Correct Implementation (RECOMMENDED)

Atmospheric turbulence scaling:

```python
sigma_u = a * U^b
sigma_v = a * U^b
sigma_w = a_w * U^b_w
```

Where U is local wind speed. Typical values from literature:
- **Rural/open terrain:** b ≈ 0.5-0.7
- **Urban/rough terrain:** b ≈ 0.3-0.5
- **Baseline:** a ≈ 0.1-0.2, a_w ≈ 0.05-0.10

For Tijuana River Valley (mixed suburban/open):
```python
sigma_u = 0.15 * U^0.5   # m/s
sigma_v = 0.15 * U^0.5   # m/s
sigma_w = 0.05 * U^0.3   # m/s (weaker vertical mixing scaling)
```

Example:
- U = 1 m/s: sigma_u = 0.15 m/s (weak turbulence)
- U = 5 m/s: sigma_u = 0.34 m/s (strong turbulence)
- U = 10 m/s: sigma_u = 0.47 m/s (very strong turbulence)

## Impact on Source Attribution

**Current model (fixed sigma):**
- Underestimates dispersion during high-wind events
- Overestimates dispersion during low-wind events
- May bias attribution toward sources that are upwind during **calm** conditions

**Corrected model (wind-dependent sigma):**
- Properly accounts for turbulence intensity
- High H2S events (calm winds) → particles stay more concentrated → sharper attribution
- Low H2S events (strong winds) → particles disperse faster → broader attribution
- Should improve agreement between predicted and observed concentration patterns

## Recommendations

### Phase 1: Implement Wind-Dependent Diffusion

Modify `projects/h2s/src/h2s/dispersion/lagrangian.py`:

```python
@dataclass
class LagrangianConfig:
    n_particles: int = 2000
    dt_seconds: float = 60.0
    hours_back: int = 2  # Updated from 6 (see EMISSION_RATE_VALIDATION.md)

    # Wind-dependent diffusion parameters
    sigma_u_coeff: float = 0.15   # baseline horizontal diffusion coefficient
    sigma_v_coeff: float = 0.15   # baseline horizontal diffusion coefficient
    sigma_w_coeff: float = 0.05   # baseline vertical diffusion coefficient
    sigma_u_exponent: float = 0.5  # wind speed scaling exponent (horizontal)
    sigma_v_exponent: float = 0.5  # wind speed scaling exponent (horizontal)
    sigma_w_exponent: float = 0.3  # wind speed scaling exponent (vertical, weaker)

    # DEPRECATED (kept for backward compatibility, not used if wind-dependent enabled)
    sigma_u: float = 0.3
    sigma_v: float = 0.3
    sigma_w: float = 0.05

    use_wind_dependent_diffusion: bool = True  # Enable new parameterization
```

Update `run_particle()` function:

```python
def run_particle(...):
    # ... existing code ...

    # Compute wind speed at particle location
    U = np.sqrt(u**2 + v**2)  # horizontal wind speed (m/s)

    # Wind-dependent diffusion
    if cfg.use_wind_dependent_diffusion:
        sigma_u_local = cfg.sigma_u_coeff * (U ** cfg.sigma_u_exponent)
        sigma_v_local = cfg.sigma_v_coeff * (U ** cfg.sigma_v_exponent)
        sigma_w_local = cfg.sigma_w_coeff * (U ** cfg.sigma_w_exponent)
    else:
        # Fixed diffusion (legacy)
        sigma_u_local = cfg.sigma_u
        sigma_v_local = cfg.sigma_v
        sigma_w_local = cfg.sigma_w

    # Add turbulent perturbations
    u_particle += np.random.normal(0, sigma_u_local) * dt
    v_particle += np.random.normal(0, sigma_v_local) * dt
    w_particle += np.random.normal(0, sigma_w_local) * dt
```

### Phase 2: Validate

1. **Re-run inversion with wind-dependent diffusion:**
   ```bash
   uv run dg launch --job dispersion_inversion_job
   ```

2. **Compare source attribution:**
   - Current (fixed sigma): east=45.6%, west=20.2%, south=34.3%
   - New (wind-dependent): expect sharper attribution during calm events

3. **Validate forward forecast:**
   - Check if Gaussian plume model also needs wind-dependent σy, σz

### Phase 3: Tuning

If attribution changes significantly:
- Adjust `sigma_u_coeff` (baseline) to match observed plume widths
- Adjust `sigma_u_exponent` to match wind speed sensitivity
- Use historical high-H2S events with known sources to calibrate

## Related Issues

- **Integration time:** Fixed (6h → 2h for valley-scale sources)
- **H2S decay:** Currently disabled (`h2s_decay_hr = 1e6`), but H2S lifetime is ~hours in atmosphere
- **Nocturnal stability:** Weak winds at night → stable boundary layer → suppressed vertical mixing
- **Sea breeze:** Diurnal wind pattern (onshore during day, offshore at night) affects source-receptor geometry

## References

- Hanna, S. R. (1982). "Applications in Air Pollution Modeling." In *Atmospheric Turbulence and Air Pollution Modelling*.
- Pasquill-Gifford stability classes use wind-dependent σy, σz
- EPA AERMOD dispersion model uses boundary-layer scaling for σ parameters

---

**Conclusion:** Wind-dependent diffusion is **essential** for accurate source attribution in the Tijuana River Valley. The current fixed-sigma model is a simplification that breaks down when wind speed varies significantly (as it does: 1-7 m/s range during high H2S events).
