# Proposed pyproject.toml changes (Reviewer draft, 2026-10-03). Builder applies.

Against `dev` at 9d5357c:

1. `authors`: replace "Your Name / your.email@example.com" with Hao Chen and Dayuan Tan (confirm emails with the user).
2. `[project.urls]`: replace all `github.com/yourusername/dataset_RFSS` with `github.com/chenhao1umbc/dataset_RFSS`;
   drop the `readthedocs.io` Documentation URL unless docs exist.
3. `classifiers`: change "License :: OSI Approved :: MIT License" to a CC BY 4.0-consistent choice (or remove the License classifier).
   `license = {text = "CC-BY-4.0"}` stays. Note CC BY 4.0 is a data license; decide the code license (for example MIT or Apache-2.0)
   and state both in the README and in a `LICENSE` file.
4. `requires-python = ">=3.13"` is stricter than needed. Test on 3.11 and 3.12 and lower the bound if the tests pass.
5. `addopts` includes `--cov=src`, which fails when `pytest-cov` is not installed. Either move `pytest-cov` into the
   default dev group (already in `[dependency-groups]`) and document `uv sync`, or drop `--cov` from `addopts`.
6. The `dependencies` list pulls in `nbstripout`, `pypdf`, `plotly`, `seaborn`, `hdf5storage`, `loguru` and torchvision/torchaudio.
   Keep only what `src/` imports; move the rest to optional groups. (Reviewer has not audited imports; Builder, please run a quick `grep`.)
7. `[tool.black]`/`mypy` target `py313`; align with item 4.
