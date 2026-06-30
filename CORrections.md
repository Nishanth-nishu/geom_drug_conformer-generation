# Architecture Corrections: Unscaled Coordinate DDPM Explosion

## 1. The Symptom
During the evaluation of the `v2_confs10` and `v2_confs5` models, the generated structures exhibited massive geometric distortions:
- **MAT-R** (Matching Recall): `~70 to 80 Å`
- **COV-R** (Coverage Recall @ 0.5Å): `0.0%`

Since normal drug-like molecules have an average diameter of 5–15 Å, an MAT-R of 70+ Å meant that the atoms were literally exploding outward into space. Despite this, the training loss decreased smoothly (from `~656` down to `~416`), indicating that the neural network was learning the score function properly.

The issue was strictly localized to the **inference/sampling logic** inside `dual_encoder_diffusion.py`.

---

## 2. The Root Cause: Scaled vs. Unscaled Coordinate Spaces

Diffusion models can parameterize the forward noising process in two ways:

### Standard DDPM (Scaled Coordinates)
Standard DDPMs (like Ho et al. 2020) gradually shrink the coordinates toward the origin so that the variance remains close to $1.0$. The forward process is:
$$ x_t^{scaled} = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$

### GeoDiff (Unscaled Coordinates)
GeoDiff, however, operates on **unscaled** molecular coordinates to preserve SO(3) translation invariance. It simply adds noise without shrinking the original coordinates:
$$ x_t^{unscaled} = \frac{x_t^{scaled}}{\sqrt{\bar{\alpha}_t}} = x_0 + \sigma_t \epsilon $$
where $\sigma_t = \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}}$.

### The Bug
Our sampling loops (`ddim_sample` and `energy_guided_ddim_sample`) correctly initialized `pos` in the **unscaled** space:
$$ \text{pos}_{T} \sim \mathcal{N}(0, \sigma_T^2 I) $$
However, the code then blindly applied the **scaled** DDPM update equations to these unscaled coordinates to predict $x_0$:
```python
# BUGGY CODE:
pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
```
Here, `at` is $\bar{\alpha}_t$. At $t=T$ (step 1), $\bar{\alpha}_T$ is extremely small (around $10^{-5}$). 
Because `pos` was already unscaled, computing `pos / sqrt(1e-5)` effectively multiplied every atomic coordinate by $\approx 316$! 

The atoms were instantly ejected 100+ Ångströms apart. The subsequent DDPM update steps tried to denoise this, but the coordinates were already permanently blown apart.

---

## 3. The Mathematical Fix

To fix this, we must map the standard DDPM posterior mean $\tilde{\mu}_t$ into the unscaled coordinate space.

### Step A: Predict $x_0$
Since our coordinates are unscaled ($x_t = x_0 + \sigma_t \epsilon$), we simply subtract the predicted noise:
$$ x_0 = x_t - \sigma_t \epsilon $$
**Code Fix:**
```python
sigma_t = (1.0 / at - 1.0).sqrt()
pos0_from_e = pos - sigma_t * e
```

### Step B: Unscaled Posterior Mean
The standard scaled DDPM posterior mean is:
$$ \tilde{\mu}_t^{scaled} = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t^{scaled} $$
To find the unscaled mean, we divide both sides by $\sqrt{\bar{\alpha}_{t-1}}$ and substitute $x_t^{scaled} = \sqrt{\bar{\alpha}_t} x_t^{unscaled}$:
$$ \tilde{\mu}_t^{unscaled} = \frac{\tilde{\mu}_t^{scaled}}{\sqrt{\bar{\alpha}_{t-1}}} = \frac{\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}\sqrt{\bar{\alpha}_t}(1-\bar{\alpha}_{t-1})}{\sqrt{\bar{\alpha}_{t-1}}(1-\bar{\alpha}_t)} x_t^{unscaled} $$
Since $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$, the second term simplifies cleanly:
$$ \tilde{\mu}_t^{unscaled} = c_1 x_0 + c_2 x_t^{unscaled} $$
Where:
- $c_1 = \frac{\beta_t}{1-\bar{\alpha}_t}$
- $c_2 = \frac{\alpha_t (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} = \frac{(1 - \beta_t)(1 - \bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$

**Code Fix:**
```python
c1 = beta_t / (1.0 - at).clamp(min=1e-8)
c2 = (1.0 - beta_t) * (1.0 - at_next) / (1.0 - at).clamp(min=1e-8)
mean_eps = c1 * pos0_from_e + c2 * pos
```

### Step C: Unscaled Posterior Variance
The standard DDPM variance is $\tilde{\beta}_t = \beta_t \frac{1 - \bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}$.
Because variance scales quadratically, the variance for the unscaled coordinate $x_{t-1}$ must be divided by $\bar{\alpha}_{t-1}$:
$$ \tilde{\beta}_t^{unscaled} = \frac{\tilde{\beta}_t}{\bar{\alpha}_{t-1}} = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{\bar{\alpha}_{t-1} (1-\bar{\alpha}_t)} $$

**Code Fix:**
```python
var_eps = beta_t * (1.0 - at_next) / ((1.0 - at).clamp(min=1e-8) * at_next.clamp(min=1e-8))
noise = torch.randn_like(pos)
pos = mean_eps + mask * torch.sqrt(var_eps) * noise
```

---

## 4. Conclusion

By substituting the mathematically exact unscaled formulas into `ddim_sample` and `energy_guided_ddim_sample`, the model no longer improperly inflates the coordinate sizes at generation time. 

The generated atoms will now safely condense down into stable geometric conformations matching the distances predicted by the properly converged dual-encoder network.
