# Releasing vllm-pspkv

This document outlines the release process for the `vllm-pspkv` fork.

## Versioning

- **Major (X)**: Breaking changes or architectural updates.
- **Minor (Y)**: New features, backward‑compatible improvements.
- **Patch (Z)**: Bug fixes, performance patches, or critical security updates.

## Release Branches & Tags

- Each release is built from a branch: `releases/vX.Y.Z`.
- Release candidates are tagged as `vX.Y.Z-rcN`.
- Final release is tagged as `vX.Y.Z`.

## Validation Checklist

Before finalising a release, run these checks:

- [ ] Build passes with GCC 12 and CUDA 12.4+.
- [ ] Unit tests pass: `./run.sh --test-unit`
- [ ] Integration tests pass: `./run.sh --test-integration`
- [ ] Manual smoke test with a small model (e.g., `facebook/opt-125m`).

## Release Steps

1. Rebase onto the latest upstream vLLM tag (if needed).
2. Update version in `setup.py`.
3. Update `CHANGELOG.md` with changes.
4. Create release branch: `git checkout -b releases/vX.Y.Z`.
5. Tag RC: `git tag vX.Y.Z-rc1 && git push origin vX.Y.Z-rc1`.
6. Run validation. Fix issues and create new RCs (`-rc2`, etc.) if needed.
7. Tag final: `git tag vX.Y.Z && git push origin vX.Y.Z`.
8. Create a GitHub Release with notes and attach the wheel (if any).
