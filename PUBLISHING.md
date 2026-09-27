# Publishing to PyPI

The distribution is named **`minimon-jax`** (`minimon` is taken on PyPI); the import
name is still `minimon`.

## 1. Bump the version

The version lives in one place, `src/minimon/__init__.py`:

```python
__version__ = "0.1.1"
```

`pyproject.toml` reads it from there (`[tool.hatch.version]`), and the value is
available at runtime as `minimon.__version__`. Add an entry to `CHANGELOG.md`.

## 2. Check and build

```bash
pip install -e ".[dev]"
ruff check src tests examples
pytest
rm -rf dist && python -m build     # -> dist/minimon_jax-X.Y.Z{.tar.gz,-py3-none-any.whl}
twine check dist/*
```

## 3a. Publish manually (simplest)

1. Create accounts on https://pypi.org and https://test.pypi.org (enable 2FA).
2. Create an API token under *Account settings → API tokens* (for the first upload the
   scope must be "entire account"; afterwards you can scope it to `minimon-jax`).
3. Dry run on TestPyPI:

   ```bash
   twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple minimon-jax
   ```

4. Real upload:

   ```bash
   twine upload dist/*        # username: __token__   password: pypi-...
   ```

   To avoid pasting the token each time, put it in `~/.pypirc`:

   ```ini
   [pypi]
   username = __token__
   password = pypi-XXXXXXXX
   ```

PyPI never lets you re-upload the same version, so bump the version for every release.

## 3b. Publish from GitHub (automated)

`.github/workflows/publish.yml` builds and uploads whenever a `v*` tag is pushed, using
PyPI *trusted publishing* (no tokens stored in GitHub).

One-time setup:

1. Push the repo to GitHub and fix the `[project.urls]` in `pyproject.toml`.
2. On PyPI: *Your account → Publishing → Add a new pending publisher*
   - PyPI project name: `minimon-jax`
   - Owner / repository: your GitHub user and repo
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. On GitHub: *Settings → Environments → New environment* named `pypi`.

Each release:

```bash
git commit -am "Release 0.1.1"
git tag v0.1.1
git push && git push --tags
```

Make sure the tag matches `__version__`.
