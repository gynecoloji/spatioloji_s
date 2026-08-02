# Releasing

Versioning, the changelog, and GitHub Releases are automated with
[release-please](https://github.com/googleapis/release-please). Archiving each
release and minting a DOI is handled by [Zenodo](https://zenodo.org). This
document covers the one-time setup and the day-to-day flow.

---

## How a release happens

1. You merge Conventional Commits into `dev` (the default branch).
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

These steps have **not** been done yet. Do them in order.

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

- **Wait for the first release PR.** Land any `feat:`/`fix:` commit on `dev`,
  then merge the release PR that appears.
- **Seed it immediately.** Create a GitHub Release manually against the existing
  `v0.3.0` tag (Releases → Draft a new release → choose tag `v0.3.0` → Publish).
  This gives you a DOI now without waiting for new work.

### 4. Paste the DOI into the repository

After the first archive, Zenodo shows two DOIs. You want the **concept DOI** —
the one described as *"Cite all versions? … represents all versions"* — because
it always resolves to the newest release.

Update three places:

1. **`README.md`** — uncomment the DOI badge near the top and replace
   `XXXXXXXX` with the concept DOI's numeric suffix. The badge image URL is
   already correct (`1123073729` is this repository's GitHub ID).
2. **`CITATION.cff`** — uncomment the `identifiers:` block at the bottom and
   fill in the same DOI.
3. Commit as `docs: add Zenodo DOI badge and citation identifier`.

---

## PyPI

Publishing to PyPI is still **manual** and is not wired into the release. After
a release PR merges:

```bash
git checkout dev && git pull
python -m build
python -m twine upload dist/*
```

If you want this automated, add a workflow triggered on `release: published`
that uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/);
it needs a matching publisher configured on the PyPI project first.

---

## Checking the setup locally

```bash
python -c "import json,pathlib; json.loads(pathlib.Path('release-please-config.json').read_text()); json.loads(pathlib.Path('.release-please-manifest.json').read_text()); print('release-please config OK')"
python -c "import yaml,pathlib; d=yaml.safe_load(pathlib.Path('CITATION.cff').read_text()); print('CITATION.cff OK —', d['version'], d['date-released'])"
```

The versions in `.release-please-manifest.json`, `pyproject.toml`,
`src/spatioloji_s/__init__.py`, and `CITATION.cff` must all agree. They are kept
in sync automatically after the first release-please run.
