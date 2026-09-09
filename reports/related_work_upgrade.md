# Related-work upgrade

## 1. Traditional inverse fitting and grid search

The current bibliography contains no dedicated grid-search or nearest-template inversion reference. Position the new nearest-physical-model experiment immediately before the ML comparison as an explicit catalog-matching baseline. Add one established domain-specific reference only after the authors select a directly comparable black-hole inverse-fitting study.

## 2. Fisher, Jacobian, and local identifiability

The manuscript already develops Fisher/Jacobian identifiability internally, but its present bibliography does not cite a general inverse-problem identifiability reference. Position the local pseudoinverse baseline after the conditioning analysis and distinguish local linear inversion from global predictive recovery. A general identifiability reference and a gravitational-wave Fisher-method reference are missing comparison points; they should be verified before addition.

## 3. Simulation-based inference and neural posterior methods

No simulation-based-inference or neural-posterior paper is currently cited. Add a short forward-looking paragraph in limitations: the present work estimates point predictions and conformal intervals, not a likelihood-free posterior. Do not imply a comparison until such a method is actually run.

## 4. Robustness and uncertainty under distribution shift

The existing bibliography cites the estimator origins (Breiman 2001; Friedman 2001) and scikit-learn (Pedregosa et al. 2011), but not conformal prediction or distribution-shift calibration. Position verified conformal and shift references alongside the empirical-coverage limitations. Keep the current language that extrapolation coverage is empirical rather than guaranteed.

## Existing relevant citations

- `breiman2001random`: Random Forests.
- `friedman2001greedy`: gradient boosting.
- `pedregosa2011scikit`: software implementation.
- `cardoso2009geodesic`, `kiselev2003quintessence`, and `palominoylla2026ringdown`: physical forward-model context, not traditional inverse baselines.

No references have been fabricated or automatically inserted into the manuscript.

