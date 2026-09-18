# Contributing to srt-slurm

Thank you for your interest in contributing to srt-slurm.

## Developer Certificate of Origin (DCO)

By contributing to this project, you agree to the Developer Certificate of Origin
(DCO) Version 1.1. This certifies that you have the right to submit your
contribution under the open source license used by the project.

The full DCO text is available at https://developercertificate.org/ and is
reproduced below:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### How to Sign Off

Add a `Signed-off-by` trailer to each commit message:

```
git commit -s -m "Your commit message"
```

This produces a commit message footer like:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE).

Each new source file must include the SPDX license header:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## Development Setup

See [CLAUDE.md](CLAUDE.md) for development environment setup and coding conventions.

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run lint + tests
make check
```

## Pull Request Guidelines

1. Ensure all tests pass (`make check`)
2. Add tests for new significant features (see `CLAUDE.md` testing section)
3. Follow existing code style (ruff enforced)
4. Include SPDX headers in all new source files
5. Sign off all commits with `git commit -s`
6. Give the PR a conventional-commit title (see below); `pr-title.yaml` (amannn/action-semantic-pull-request) rejects anything else

## Versioning

There is one version, the git tag, and everything else is derived from it.

- Every merge to `main` cuts a GitHub release. `release.yaml` reads the highest existing release, bumps it from the merged PR title, creates the tag on the merge commit **first**, and only then builds the wheel and the binaries from that tag.
- The PR title decides the bump (conventional commits):

  | Title | Bump | Example |
  |---|---|---|
  | `<type>!: ...` or a body with `BREAKING CHANGE:` | MAJOR | `feat!: drop the v1 recipe layout` |
  | `feat: ...` / `feat(scope): ...` | MINOR | `feat(roles): add nodes: colocate` |
  | `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert` | PATCH | `fix(sglang): pass the bootstrap port` |

- The Python package version is the tag (`hatch-vcs`, `pyproject.toml`); `srtctl --version` prints it with the commit and the recipe-schema and lockfile versions it speaks. A checkout between tags reports `2.0.0.post3+g<sha>`; a tree without git metadata reports `0.0.0+unknown`. Do not hand-edit a version anywhere.
- The Rust binaries (`tachometer-scraper`, `cpu-power-exporter`) print the same tag from `--version`; the release build passes it as `SRTCTL_RELEASE_VERSION`. The `version` in `Cargo.toml` is a baseline for local builds only.
- Every run stamps `srtctl_version` and `srtctl_commit` into `recipe.lock.yaml` and the resource snapshot, so an output directory says exactly which srtctl produced it.
- Format versions are separate from the package version on purpose: `schema:` in a recipe (`SUPPORTED_SCHEMA_VERSIONS`), the lockfile version, and the status API spec. Each release's notes list which of them that release reads and writes.
- To publish a version by hand (a MAJOR bump, or to correct history), create the release with `gh release create vX.Y.Z --target <sha>`; the next merge continues from it.
