
## Publishing (OIDC: TestPyPI then PyPI)

This repository is configured for **Trusted Publishing** with GitHub OIDC via `.github/workflows/publish.yml`.

Release flow:

1. Create a GitHub Release (`published`) with the desired version tag (for example `v0.2.6`).
2. GitHub Actions builds the package once.
3. It publishes to **TestPyPI** first.
4. If TestPyPI publish succeeds, it publishes the same artifacts to **PyPI**.

### One-time setup on package indexes

You must create a Trusted Publisher on both TestPyPI and PyPI for project `mlweather` with:

- Owner: `kap-conseil`
- Repository name: `mlweather`
- Workflow name: `publish.yml`
- Environment name:
    - `testpypi` for TestPyPI
    - `pypi` for PyPI

No API token is required when Trusted Publishing is configured correctly.