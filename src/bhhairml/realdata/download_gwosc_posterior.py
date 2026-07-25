"""Discover and download public GWOSC parameter-estimation posterior samples."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

API_ROOT = "https://gwosc.org/api/v2"


def _json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "bhhairml/0.1 public-data interface"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def discover_event(event: str) -> dict:
    """Find the newest GWOSC event version with a preferred PE data URL."""
    event = event.upper()
    candidates = []
    # API v2 event-version numbering is currently small. Probe newest first;
    # unsuccessful versions are normal and are not fatal.
    for version in range(12, 0, -1):
        uid = f"{event}-v{version}"
        parameter_url = f"{API_ROOT}/event-versions/{uid}/parameters?format=json"
        try:
            payload = _json(parameter_url)
        except HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        entries = payload.get("results", payload if isinstance(payload, list) else [])
        pe = [entry for entry in entries
              if entry.get("pipeline_type") == "pe" and entry.get("data_url")]
        if not pe:
            continue
        preferred = [entry for entry in pe if entry.get("is_preferred")]
        chosen = (preferred or pe)[0]
        candidates.append({"event": event, "event_uid": uid, "version": version,
                           "parameters_api_url": parameter_url,
                           "parameter_estimation": chosen,
                           "api_response": payload})
        break
    if not candidates:
        raise LookupError(f"No public PE posterior data URL found for {event}")
    return candidates[0]


def download_posterior(event: str = "GW150914", *, output_dir="data/real/gwosc",
                       force: bool = False) -> Path | None:
    """Download the preferred public posterior, returning None on recoverable failure."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / f"{event.upper()}_metadata.json"
    if metadata_path.exists() and not force:
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cached_path = Path(cached_metadata.get("local_path", ""))
            if cached_path.exists() and cached_path.is_file():
                cached_metadata["download_status"] = "reused existing file"
                metadata_path.write_text(json.dumps(cached_metadata, indent=2), encoding="utf-8")
                print(cached_path)
                return cached_path
        except (ValueError, OSError):
            pass
    if not force:
        cached_files = [
            path for path in destination.iterdir()
            if path.is_file() and event.upper() in path.name.upper()
            and path.suffix.lower() in {".h5", ".hdf5", ".hdf", ".csv", ".json"}
            and path != metadata_path
        ]
        if len(cached_files) == 1:
            local = cached_files[0]
            recovered = {"event": event.upper(), "local_path": str(local.resolve()),
                         "download_status": "recovered cached file; API metadata unavailable"}
            metadata_path.write_text(json.dumps(recovered, indent=2), encoding="utf-8")
            print(local)
            return local
    try:
        metadata = discover_event(event)
        url = metadata["parameter_estimation"]["data_url"]
        filename = Path(urlparse(url).path).name or f"{event.upper()}_posterior.h5"
        local = destination / filename
        metadata.update({"download_url": url, "local_path": str(local.resolve())})
        if local.exists() and not force:
            metadata["download_status"] = "reused existing file"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(local)
            return local
        request = Request(url, headers={"User-Agent": "bhhairml/0.1 public-data interface"})
        temporary = local.with_suffix(local.suffix + ".part")
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            metadata["content_type"] = response.headers.get("Content-Type")
            metadata["content_length_header"] = response.headers.get("Content-Length")
        temporary.replace(local)
        metadata["download_status"] = "downloaded"
        metadata["downloaded_bytes"] = local.stat().st_size
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(local)
        return local
    except (HTTPError, URLError, TimeoutError, LookupError, OSError, ValueError) as exc:
        failure = {"event": event.upper(), "download_status": "failed",
                   "error": f"{type(exc).__name__}: {exc}",
                   "manual_instructions": [
                       f"Open https://gwosc.org/eventapi/html/event/{event.upper()}/",
                       "Follow the preferred Parameter Estimation / Posterior Samples data link.",
                       f"Save the HDF5/CSV/JSON file under {destination.resolve()}.",
                       "Run posterior_constraints with --posterior PATH or rerun run_realdata_demo."
                   ]}
        metadata_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"Automatic GWOSC posterior download failed: {exc}")
        for instruction in failure["manual_instructions"]:
            print(f"- {instruction}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="GW150914")
    parser.add_argument("--output-dir", default="data/real/gwosc")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_posterior(args.event, output_dir=args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
