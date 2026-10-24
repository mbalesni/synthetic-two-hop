# /// script
# requires-python = ">=3.11"
# dependencies = ["pip-audit==2.10.1"]
# ///
"""Audit every registry package/version in uv.lock, including alternate variants.

Run with `uv run --script scripts/audit_lock.py`. No advisories are suppressed;
exit status 1 means vulnerabilities remain, and 2 means the audit failed.
"""

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def main():
    lock = Path(__file__).resolve().parents[1] / "uv.lock"
    packages = tomllib.loads(lock.read_text())["package"]
    variants = {}
    for package in packages:
        if "registry" in package.get("source", {}):
            variants.setdefault(package["name"], set()).add(package["version"])
    batches = []
    for name, versions in sorted(variants.items()):
        for index, version in enumerate(sorted(versions)):
            if index == len(batches):
                batches.append([])
            batches[index].append(f"{name}=={version}")

    vulnerable = False
    audited = 0
    with tempfile.TemporaryDirectory(prefix="lock-audit-") as directory:
        for index, batch in enumerate(batches):
            requirements = Path(directory) / f"requirements-{index}.txt"
            requirements.write_text("\n".join(batch) + "\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--no-deps",
                    "--disable-pip",
                    "--progress-spinner",
                    "off",
                    "--format",
                    "json",
                    "-r",
                    str(requirements),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode not in (0, 1):
                print(result.stderr, file=sys.stderr)
                return 2
            try:
                dependencies = json.loads(result.stdout)["dependencies"]
            except (ValueError, KeyError):
                print(result.stdout + result.stderr, file=sys.stderr)
                return 2
            if len(dependencies) != len(batch) or any("skip_reason" in d for d in dependencies):
                print("Incomplete audit:", dependencies, file=sys.stderr)
                return 2
            audited += len(dependencies)
            for dependency in dependencies:
                if dependency["vulns"]:
                    vulnerable = True
                    print(
                        json.dumps(
                            {
                                "name": dependency["name"],
                                "version": dependency["version"],
                                "vulns": [
                                    {
                                        key: value
                                        for key, value in vuln.items()
                                        if key != "description"
                                    }
                                    for vuln in dependency["vulns"]
                                ],
                            },
                            indent=2,
                        )
                    )
    print(f"Audited all {audited} registry package/version entries; no advisories ignored.")
    return int(vulnerable)


if __name__ == "__main__":
    raise SystemExit(main())
