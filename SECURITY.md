# Dependency security

Python 3.11 or newer is required. Install the reviewed dependency versions with
`uv sync --locked --python python3.11`. Existing Python 3.10 environments must
be recreated with a supported interpreter.

## October 24, 2026 review

The October notifications still apply to the September lockfile on `main`
(`6fa771a5`). At the start of the review, PR #3 was the latest PR and neither
existing branch contained a newer fix.
The update uses the following first fixed versions, without a blanket upgrade:

| Dependency | Previous | Updated | Advisory / reason |
| --- | --- | --- | --- |
| certifi | 2026.6.17 | 2026.10.19 | [GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2), compromised emSign CA trust |
| GitPython | 3.1.60 | 3.2.1 | [CVE-2026-114145](https://github.com/advisories/GHSA-83vg-56qc-22m7), submodule RCE; also fixes [submodule path traversal](https://github.com/advisories/GHSA-59cr-6r3x-644w) |
| virtualenv (dev) | 21.5.1 | 21.14.13 | Activation, creation, cache and seeding advisories; includes the [21.14.9–21.14.12 Bash regression fix](https://github.com/advisories/GHSA-cx6f-vjhg-g3cc) |
| aiohttp | 3.14.3 | 3.14.4 | Eight advisories, including [CVE-2026-111066](https://github.com/advisories/GHSA-9xw5-w5jj-pvcg), multipart file descriptor exhaustion |
| fsspec | 2025.9.0 | 2026.6.0 | [CVE-2026-104851](https://github.com/advisories/GHSA-27vj-qcqg-25rc), reference filesystem template injection |
| langgraph-sdk | 0.4.2 | 0.4.4 | [CVE-2026-104873](https://github.com/advisories/GHSA-fvww-7h3r-vfhp), ignored authorization actions |
| LiteLLM | 1.89.2 | 1.89.7 | [CVE-2026-84377](https://github.com/advisories/GHSA-3cv6-jpf6-8222), proxy SSRF/credential disclosure; stay on the 1.89 patch line |
| multidict | 6.7.1 | 6.9.1 | [CVE-2026-104874](https://github.com/advisories/GHSA-54p9-h82j-f925), reference leak |
| setuptools | 83.0.0 | 84.1.0 | [GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w), Windows legacy wheel path escape; also raise the isolated build floor |
| Tornado | 6.5.8 | 6.5.9 | [Query count DoS](https://github.com/advisories/GHSA-3hv7-mjh2-fv65), [response size DoS](https://github.com/advisories/GHSA-chx6-46f5-w4vp), [static symlink traversal](https://github.com/advisories/GHSA-c2m8-h5v5-343r) |
| urllib3 | 2.7.0 | 2.8.0 | [Proxy TLS](https://github.com/advisories/GHSA-8988-9cw3-xx77), [Deflate loop](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g), [chunk header buffering](https://github.com/advisories/GHSA-vxq7-64xx-v4gw) |
| filelock | 3.29.4 | 4.2.1 | Three advisories fixed; Python >=3.11 is now required |

The associated compatibility updates are s3fs 2026.6.0 (matches fsspec),
yarl 1.25.1 (required by aiohttp), and packaging 26.3 and python-discovery 1.6.1
(required by virtualenv).
The existing datasets 5.0.1 allows fsspec 2026.6.0, so no datasets update is needed.
The direct LiteLLM pin and build requirement change; uv constraints preserve
security floors without making transitive packages new runtime dependencies.
PyTorch, Transformers, Accelerate, TRL and the other direct pins stay unchanged.

These are affected installed versions, not evidence that each vulnerable code
path is exercised by the experiments. Certifi affects outbound TLS trust.
GitPython arrives through W&B, and virtualenv through the dev pre-commit extra.
No direct aiohttp server or submodule operations were found in repository Python
code. The aiohttp multipart issue specifically requires a handler calling
`Request.post()`; client-side cookie/connector issues are separate. Other findings
also have feature/platform preconditions (for example, the LiteLLM proxy and
Windows setuptools installation). They are still patched rather than dismissed.

Rebuild existing environments and images from the updated lockfile. Previously
generated virtualenv activation scripts need recreation to pick up their fixes.
This repository change does not update globally installed tools or deployed images.

### Python minimum and filelock remediation

filelock 4.2.1 requires Python >=3.11. The project now requires that interpreter
minimum and unconditionally constrains filelock to >=4.2.1. The vulnerable 3.29.4
fallback and all Python 3.10-only lockfile entries have been removed. This fixes:

- [CVE-2026-112940](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8): SQLite-backed ReadWriteLock can follow attacker-controlled paths/side files.
- [CVE-2026-112913](https://github.com/advisories/GHSA-9vhv-jxww-xfm5): Windows SQLite side-file interference.
- [CVE-2026-112907](https://github.com/advisories/GHSA-567q-5qvw-pf92): Windows soft-lock waits can exceed timeouts under an attacker-held oplock.

Python 3.10 is no longer supported. Recreate existing environments with Python
3.11 or newer and this lockfile; changing the repository does not repair an
older environment that still contains filelock 3.29.4.

## Outstanding upstream advisory

Rechecked October 24, 2026: Accelerate 1.14.0 is affected by
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

The latest 1.15.0 wheel was checked against PyPI's SHA256
`97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec`;
`accelerate/utils/modeling.py:1944` still joins index values to the checkpoint
folder without containment or special-file validation. The advisory has no
first patched release. An upgrade to 1.15.0 is therefore not a verified fix.

## Validation

Revalidated October 24 on macOS arm64 with Python 3.11.10 after raising the
minimum: locked installation, dependency compatibility, both derangement tests,
and the offline smoke script passed. All 209 registry package/version entries
were audited; only the unresolved Accelerate advisory remains. Lock consistency,
script lint/format, and diff whitespace checks passed. Explicitly requesting
Python 3.10 is rejected as incompatible with the project requirement.

Run from the repository root with Python 3.11 or newer:

```sh
DS_BUILD_OPS=0 uv sync --locked --all-extras --python python3.11
uv lock --check
uv pip check
uv run --locked --all-extras python -m pytest latent_reasoning/datagen/synthetic_spouses/test_derangement.py -q
uv run --locked --all-extras python scripts/security_smoke.py
uv run --script scripts/audit_lock.py
git diff --check
```

The smoke test is offline and covers dependency imports, a CPU tiny-Llama
forward/backward/optimizer step, dataset persistence, file locking, a Git index,
and virtualenv creation with embedded seed wheels. Install the dev extra for it.

The audit includes every registry package/version in the universal lockfile,
including dev and alternate Python/platform versions. It intentionally returns
1 while known vulnerabilities remain; nothing is ignored. The expected residual
finding is Accelerate. GitHub advisory metadata was also checked because PyPI/OSV
ingestion can lag newly reviewed advisories. There is no repository-defined test
CI; GitHub exposes a managed Dependency Graph workflow.

The GitHub notifications were verified, including aiohttp alert #192. Direct
Dependabot alert-state access was unavailable (public page 404; local GitHub CLI
401; connector lacks the alerts endpoint), so this review does not claim that
GitHub has closed alerts. No GPU/multi-GPU training, gated model downloads or
paid external API evaluations are covered by these checks.
