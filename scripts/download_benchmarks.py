"""Download public JSPLIB data, pin commit and record SHA-256 provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULTS = ["ft06", "ft10", "ft20", "la01", "la16", "la31", "abz5", "ta51", "ta71"]


def get(url):
    with urlopen(Request(url, headers={"User-Agent": "mas-jssp-research"}), timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--instances", nargs="+", default=DEFAULTS)
    parser.add_argument("--revision", help="JSPLIB commit SHA; default resolves master once")
    args = parser.parse_args()
    revision = args.revision or json.loads(get(
        "https://api.github.com/repos/tamy0612/JSPLIB/commits/master"))["sha"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a 40-character commit SHA")
    base = f"https://raw.githubusercontent.com/tamy0612/JSPLIB/{revision}/"
    metadata = {item["name"]: item for item in json.loads(get(base + "instances.json"))}
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.instances:
        item = metadata[name]
        raw = get(base + item["path"])
        (args.output / f"{name}.txt").write_bytes(raw)
        records.append({**item, "file": f"{name}.txt",
                        "url": base + item["path"],
                        "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"{name}: {item['jobs']} x {item['machines']}", flush=True)
    manifest = {"repository": "https://github.com/tamy0612/JSPLIB",
                "revision": revision, "instances": records}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
