# Releasing

Versioning, the changelog, and GitHub Releases are automated with
[release-please](https://github.com/googleapis/release-please). Archiving each
release and minting a DOI is handled by [Zenodo](https://zenodo.org). This
document covers the one-time setup and the day-to-day flow.

---

## How a release happens

1. You merge Conventional Commits into `main` (the default branch).
2. The `release-please` workflow opens — or updates — a **release PR** titled
   something like `chore(main): release 0.4.0`. It contains only version and
   changelog edits.
3. When you merge that release PR, release-please:
   - tags `v0.4.0` and publishes a **GitHub Release**,
   - writes the new `CHANGELOG.md` section,
   - bumps the version in `pyproject.toml`, `src/spatioloji_s/__init__.py`,
     and `CITATION.cff` (both `version:` and `date-released:`).
4. The GitHub Release fires the Zenodo webhook, which archives the tarball and
   mints a **new version DOI** under the existing **concept DOI**.
5. You upload to PyPI with one manual command (see [PyPI](#pypi)):
   `gh workflow run publish.yml --ref vX.Y.Z -f tag=vX.Y.Z`

You never edit `CHANGELOG.md` or a version number by hand.

### Which commits cause which bump

Two things are decided separately: **what size bump** you get, and **what shows
up in the changelog**.

The bump is driven by these prefixes:

| Prefix | Bump |
|---|---|
| `feat!: …`, or a `BREAKING CHANGE:` footer | major |
| `feat: …` | minor |
| `fix: …` | patch |

Because the project is pre-1.0, a breaking change bumps the *minor* version
(0.3.0 → 0.4.0) rather than the major — standard SemVer, and release-please
follows it.

The changelog sections come from `changelog-sections` in
`release-please-config.json`. `feat`, `fix`, `perf`, `refactor`, and `docs` are
visible and get their own headings; `test`, `build`, `ci`, and `chore` are
marked hidden and never appear.

A batch of only hidden-type commits will not produce a release worth cutting, so
if you want your work released, make sure at least one commit is a `feat:` or a
`fix:`.

---

## One-time setup

Steps 1–3 are **done**. Step 4 is partly done: the DOI badge is in `README.md`,
but `CITATION.cff` still carries the placeholder. Step 5 is outstanding.

### 1. Allow GitHub Actions to open pull requests

Repository **Settings → Actions → General → Workflow permissions** → tick
*"Allow GitHub Actions to create and approve pull requests"*.

Without this, the `release-please` workflow fails with a 403 when it tries to
open the release PR.

### 2. Enable the Zenodo integration

1. Sign in at <https://zenodo.org> with your GitHub account.
2. Go to <https://zenodo.org/account/settings/github/> and grant access to the
   `gynecoloji` account if prompted.
3. Find `gynecoloji/spatioloji_s` in the repository list and flip the toggle to
   **On**.

Zenodo only archives releases published *after* the toggle is on — the existing
`v0.1.0` and `v0.3.0` tags have no GitHub Release attached and will be ignored.

### 3. Publish a first release to mint the concept DOI

Two options:

- **Wait for the first release PR.** Land any `feat:`/`fix:` commit on `main`,
  then merge the release PR that appears.
- **Seed it immediately.** Create a GitHub Release manually against the existing
  `v0.3.0` tag (Releases → Draft a new release → choose tag `v0.3.0` → Publish).
  This gives you a DOI now without waiting for new work.

### 4. Paste the DOI into the repository

After the first archive, Zenodo shows two DOIs. You want the **concept DOI** —
the one described as *"Cite all versions? … represents all versions"* — because
it always resolves to the newest release.

The concept DOI for this project is **`10.5281/zenodo.21753918`**.

1. **`README.md`** — done; the badge is in the header block.
2. **`CITATION.cff`** — still to do. Uncomment the `identifiers:` block at the
   bottom and fill in the concept DOI above. This feeds both GitHub's "Cite this
   repository" button and the metadata Zenodo attaches to future deposits.

### 5. Register the PyPI trusted publisher

See the [PyPI](#pypi) section below. Until that is configured, the publish
workflow will run on each release and fail at the upload step.

---

## PyPI

Publishing is a deliberate manual step. After merging a release PR, run

```bash
gh workflow run publish.yml --ref vX.Y.Z -f tag=vX.Y.Z
```

`--ref vX.Y.Z` builds exactly the released commit; the `tag` input arms the
version gate below. (The upload used to be chained from release-please.yml via
`workflow_call`, but PyPI's attestation check rejects reusable workflows — the
signing certificate names the caller workflow, not `publish.yml` — which
blocked the v0.4.8 upload. A top-level dispatch matches the trusted publisher
exactly, attestations included.)

Authentication is [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/):
PyPI accepts a short-lived OIDC identity token minted by GitHub for that one
workflow. There is no API token in the repository, in Actions secrets, or on any
developer machine — nothing to leak, rotate, or revoke.

### One-time PyPI setup

On pypi.org → your project → **Manage → Publishing → Add a new publisher →
GitHub**, enter exactly:

| Field | Value |
|---|---|
| Owner | `gynecoloji` |
| Repository | `spatioloji_s` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

All four are matched literally. A mismatch makes PyPI reject the upload with an
authentication error rather than a helpful one, so check them character by
character.

### What the workflow does before uploading

Three gates, all of which must pass:

1. **Clean build** — `rm -rf build dist src/*.egg-info` first, because
   setuptools reuses a stale `build/lib` and can ship files deleted from source.
2. **`twine check`** — confirms the metadata renders on PyPI.
3. **Install-and-import** — installs the built wheel into an empty virtualenv
   with only its declared dependencies and imports it. This is the gate that
   catches a module importing something missing from `pyproject.toml`.

On a release it also asserts the built `__version__` matches the tag, so a
mismatched version can never reach PyPI.

### Manual publish

`workflow_dispatch` is enabled, so you can publish from the Actions tab without
cutting a release — useful for a version tagged before this workflow existed.
To publish from your machine instead, the old path still works:

```bash
rm -rf build dist src/*.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Remember that **a version number can never be reused** on PyPI. If a bad release
goes out, you yank it and ship the next patch; you cannot re-upload.

---

## Checking the setup locally

```bash
python -c "import json,pathlib; json.loads(pathlib.Path('release-please-config.json').read_text()); json.loads(pathlib.Path('.release-please-manifest.json').read_text()); print('release-please config OK')"
python -c "import yaml,pathlib; d=yaml.safe_load(pathlib.Path('CITATION.cff').read_text()); print('CITATION.cff OK —', d['version'], d['date-released'])"
```

The versions in `.release-please-manifest.json`, `pyproject.toml`,
`src/spatioloji_s/__init__.py`, and `CITATION.cff` must all agree. They are kept
in sync automatically after the first release-please run.
