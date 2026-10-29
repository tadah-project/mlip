# Contributing to MLIP (engine library)

The full multi-repo contributor guide for the Tadah! project lives on
the umbrella repository:
<https://github.com/tadah-project/tadah.mlip/blob/v1-dev/CONTRIBUTING.md>

This file covers what is specific to the MLIP engine library. Bug
reports and feature requests on this module use the issue tracker of the
umbrella project, <https://github.com/tadah-project/tadah.mlip/issues>;
cross-cutting changes (work that touches more than one Tadah! module)
should be discussed on the umbrella project first.

## Build & test in isolation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build/tests --output-on-failure
```

In-source builds are blocked deliberately; always pick a separate
build directory.

### Dependencies

- C++17 compiler.
- [CORE](https://github.com/tadah-project/core) and
  [MODELS](https://github.com/tadah-project/models) — fetched
  automatically.
- [Boost.Iostreams](https://www.boost.org/) (linked at link time).
- [Gemmi](https://gemmi.readthedocs.io/) — fetched automatically
  (used by the CIF reader).
- LAPACK / BLAS at link time (provided by the build environment).

## Layout

| Path | Purpose |
|---|---|
| `include/tadah/mlip/` | Public headers; one per engine (`train_engine.h`, `predict_engine.h`, `data_*_engine.h`, `analysis_*_engine.h`, `load_transforms.h`, …). |
| `src/`                | Implementation translation units. |
| `tests/`              | Engine-level integration tests. |
| `external/`           | Vendored third-party headers. |
