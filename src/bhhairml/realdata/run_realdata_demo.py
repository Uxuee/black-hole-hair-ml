"""One-command public-event posterior uncertainty-scale demonstration."""
from __future__ import annotations
import argparse
from pathlib import Path
from bhhairml.realdata.download_gwosc_posterior import download_posterior
from bhhairml.realdata.posterior_constraints import run
from bhhairml.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="GW150914")
    parser.add_argument("--config", default="configs/realdata_config.yaml")
    parser.add_argument("--posterior", help="Use a local posterior file instead of downloading")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    posterior = Path(args.posterior) if args.posterior else download_posterior(
        args.event, force=args.force)
    if posterior is None:
        print("No posterior was analyzed. Supply --posterior PATH after manual download.")
        return
    config = load_yaml(args.config)
    config["event_name"] = args.event
    result = run(posterior, config)
    figures = [
        "realdata_kerr_qnm_posterior", "realdata_frequency_damping_posterior",
        "realdata_allowed_bardeen_q", "realdata_allowed_hayward_q",
        "realdata_allowed_kiselev_region"]
    print(f"event name: {args.event}")
    print(f"posterior file path: {posterior.resolve()}")
    print(f"number of samples: {len(result['samples'])}")
    print("detected columns:")
    for canonical, source in result["mapping"].items():
        print(f"  {canonical}: {source}")
    print(f"Kerr QNM method used: {result['summary']['backend']}")
    print("generated reports: reports/realdata_connection_report.md, .html, .pdf")
    print("generated figures:")
    for figure in figures:
        print(f"  reports/figures/{figure}.png and .pdf")
    print("SCIENTIFIC WARNING: this is not a black-hole-hair detection. It is an "
          "observational-scale toy tolerance comparison under leading-eikonal assumptions.")


if __name__ == "__main__":
    main()
