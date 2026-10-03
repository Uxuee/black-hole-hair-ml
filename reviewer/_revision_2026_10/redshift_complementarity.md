# Redshift complementarity check

## Scope and sources

This reviewer-only analysis uses archived physical feature tables and existing
split/model definitions. It runs no geodesic calculation and changes no
canonical artifact.

- Jacobians use the nominal 81-phase `combined_features.csv`, the paper's
  global valid-grid q95-q05 feature scaling, the same parameter spans and
  finite-difference rules, and separate analytic handling of $k=0$.
- Inverse recovery uses the archived 161-phase feature table and all 115
  existing split specifications. Existing Ringdown and Ringdown + Photon
  Geometry fold metrics are reused. Only the two previously unavailable
  redshift combinations are fitted, with the existing HGB, RF, and scaled MLP
  definitions and training-only preprocessing.
- Ordinary $w_q$ scores exclude every $k=0$ test point.

## Feature sets

| Feature set | Definition | Dimension |
|---|---|---:|
| Ringdown | $\Omega$, $\lambda$, $\Delta r$, and $r_{\rm photon}$ | 4 |
| Ringdown + Redshift | Ringdown plus mean, amplitude, reference, and sine/cosine harmonics 1-3 of redshift | 13 |
| Ringdown + Photon Geometry | Ringdown plus the same nine summaries for impact parameter, $\alpha_{\rm sky}$, and $\beta_{\rm sky}$ | 31 |
| Ringdown + Photon Geometry + Redshift | Union of the preceding photon-geometry and redshift combinations | 40 |

## Nominal 81-phase Jacobian comparison

All summaries contain 110 eligible finite-$k$ points. Ratios are computed
point by point before taking their median.

| Feature set | $\sigma_{\min}$ median [IQR] | $\kappa(J)$ median [IQR] | Median pointwise $\sigma_{\min}$ gain | Gain $>1$ | Median pointwise conditioning improvement | Conditioning improved |
|---|---:|---:|---:|---:|---:|---:|
| Ringdown | 0.278233 [0.162299, 0.418109] | 6.611166 [4.901971, 11.440288] | 1.000000 | 0.0% | 1.000000 | 0.0% |
| Ringdown + Redshift | 0.945900 [0.520140, 1.497074] | 3.610689 [2.719785, 6.052608] | 3.291017 | 100.0% | 2.272650 | 84.5% |
| Ringdown + Photon Geometry | 1.232579 [0.844332, 1.751993] | 4.073124 [2.748047, 6.303361] | 4.542865 | 100.0% | 2.281889 | 80.0% |
| Ringdown + Photon Geometry + Redshift | 1.266524 [0.896912, 1.782558] | 4.287750 [2.780263, 6.919940] | 4.748933 | 100.0% | 2.118434 | 77.3% |

The exact $k=0$ rows remain rank one with zero $\sigma_{\min}$ for all four
sets and are stored separately from finite-$k$ ratios.

## Identifiable-$w_q$ inverse recovery

The all-model values below use the manuscript hierarchy: first take a median
within each model/direction subgroup, keep the four extrapolation directions
separate, then take the median of those subgroup medians.

| Feature set | Grouped NMAE | Directional NMAE |
|---|---:|---:|
| Ringdown | 0.309275 | 0.301061 |
| Ringdown + Redshift | 0.099094 | 0.192222 |
| Ringdown + Photon Geometry | 0.086522 | 0.204771 |
| Ringdown + Photon Geometry + Redshift | 0.078426 | 0.200431 |

Model-specific medians are:

| Feature set | Protocol | HGB | RF | MLP |
|---|---|---:|---:|---:|
| Ringdown | Grouped | 0.314541 | 0.309275 | 0.072898 |
| Ringdown + Redshift | Grouped | 0.101232 | 0.099094 | 0.068044 |
| Ringdown + Photon Geometry | Grouped | 0.086522 | 0.111317 | 0.073751 |
| Ringdown + Photon Geometry + Redshift | Grouped | 0.078426 | 0.079191 | 0.073994 |
| Ringdown | Directional | 0.350208 | 0.351597 | 0.183901 |
| Ringdown + Redshift | Directional | 0.199587 | 0.202174 | 0.154536 |
| Ringdown + Photon Geometry | Directional | 0.214639 | 0.203281 | 0.189602 |
| Ringdown + Photon Geometry + Redshift | Directional | 0.194595 | 0.200636 | 0.269325 |

The two new MLP feature-set runs emitted 47 convergence warnings under the
unchanged 600-iteration configuration. Tree-family results therefore provide
the cleaner comparison, while the MLP remains visible rather than being
discarded.

## Nearest-physical-model baseline

| Feature set | Grouped NMAE | Directional NMAE |
|---|---:|---:|
| Ringdown | 0.220833 | 0.254161 |
| Ringdown + Redshift | 0.158532 | 0.193228 |
| Ringdown + Photon Geometry | 0.148148 | 0.195026 |
| Ringdown + Photon Geometry + Redshift | 0.132143 | 0.195026 |

## Interpretation

Redshift supplies **moderate genuine complementarity** relative to Ringdown:
it raises the weakest Jacobian direction by a median factor of 3.29, improves
conditioning at 84.5% of finite-$k$ points, and materially lowers grouped and
directional recovery error in both tree families and the nearest-model
baseline. This is more than additional response magnitude along an unchanged
direction.

Photon geometry nevertheless captures most of the useful complementary
direction. Adding redshift on top of Ringdown + Photon Geometry raises the
median pointwise $\sigma_{\min}$ gain only from 4.543 to 4.749, while median
conditioning becomes slightly worse (improvement relative to Ringdown falls
from 2.282 to 2.118). The triple set improves grouped tree and nearest-model
recovery. Its all-model directional NMAE improves only marginally from
0.204771 to 0.200431, while its directional MLP median is substantially worse.

## Recommendation

Do not add a new main-text result at this stage. The clean, useful conclusion
is suitable for a concise appendix sentence or reviewer response: redshift is
complementary to ringdown, but adds little robust directional information once
photon geometry is present. Any manuscript integration should foreground the
tree/nearest consistency and retain the mixed directional/MLP result rather
than describing the triple set as uniformly superior.
