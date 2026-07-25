# Optional public-event posterior interface

No real event posterior has been processed yet. This document describes the
interface and is replaced by an event-specific `realdata_connection_report.md`,
`.html`, and `.pdf` after running:

```powershell
python -m bhhairml.realdata.posterior_constraints `
  --posterior PATH_TO_POSTERIOR_FILE `
  --config configs/realdata_config.yaml
```

The branch reads local GWOSC/LVK parameter-estimation samples rather than raw
strain. It computes a Kerr remnant QNM baseline and compares posterior uncertainty
scales with toy static leading-eikonal hair shifts. These comparisons are not
black-hole-hair detections or observational modified-gravity bounds.
