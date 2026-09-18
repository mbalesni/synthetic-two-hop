# Dependency security

Install the reviewed dependency versions with `uv sync --locked`.

## Outstanding upstream advisory

As of September 18, 2026, Accelerate 1.14.0 is affected by
[CVE-2026-69112 / GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79).
Its checkpoint loaders can follow paths outside the checkpoint directory and
can block on special files referenced by a sharded checkpoint index.

Only load checkpoints from trusted sources. Do not run experiments against
untrusted local checkpoint directories or sharded checkpoint indexes. This is
an operational precaution, not a code-level fix.

The advisory lists no patched release. Accelerate 1.15.0 still joins unchecked
`weight_map` paths in `load_checkpoint_in_model`, so upgrading solely to clear
the scanner's affected-version range would not resolve the issue. Keep this
advisory visible in dependency audits until an upstream fix is verified.
