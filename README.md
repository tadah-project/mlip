[![pipeline](https://git.ecdf.ed.ac.uk/tadah/mlip/badges/develop/pipeline.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/commits/develop)
[![coverage](https://git.ecdf.ed.ac.uk/tadah/mlip/badges/develop/coverage.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/commits/develop)
[![license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![version](https://img.shields.io/badge/version-1.3.0--beta.1-blue.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/tags)

# Tadah!MLIP (engine library)

Engine library for the [Tadah!](https://tadah.readthedocs.io/) MLIP
toolchain. This is **not** the user-facing project — it is consumed
by the umbrella project
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip), which links
this library and provides the `tadah` CLI binary plus documentation.

## What lives here

The `tadah.mlip` library implements the engines that back every
`tadah` subcommand:

- **Training** — design-matrix construction, regression dispatch,
  potential-file emission.
- **Prediction** — apply a trained `pot.tadah` to datasets or
  individual structures; per-frame analytics.
- **Data engines** — dataset conversion (CASTEP / VASP / AFLOW / COD /
  CIF / dataset readers), de-duplication, merging, splitting,
  sampling, weight rebalancing.
- **Analysis engines** — basis-function and cutoff-function plots,
  descriptor calculation.
- **Properties engine** — pairwise energy curves.
- **Load-time data transforms** — `LSCALE`, `ESHIFT` (explicit /
  ATOM / DBATOM), `EFILTER`, `FFILTER`, `WDBFILE`, `WDBFILE_AUTO`,
  `EWEIGHT_TEMP`, `ZERO_COM_FORCE`, equilibrium-volume diagnostic.
  Implemented in `load_transforms.{h,cpp}`; documented in the
  *Load-time data transforms* section of the Tadah!MLIP HPO docs.

## When you use this directly

You should not need to. The user-facing entry points are:

- [Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip) — the
  `tadah` CLI (training, prediction, data tooling) with documentation
  at <https://tadah.readthedocs.io/>.
- [Tadah!LAMMPS](https://git.ecdf.ed.ac.uk/tadah/tadah.lammps) —
  `pair_style tadah` for LAMMPS.

This repository is broken out separately so that the
hyperparameter-optimisation engine
([Tadah!HPO](https://git.ecdf.ed.ac.uk/tadah/hpo)) can link the same
training/prediction code path that the CLI does.

## Build & test in isolation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build/tests --output-on-failure
```

In-source builds are blocked deliberately; always pick a separate
build directory.

### Dependencies

- C++17 compiler.
- [Tadah!CORE](https://git.ecdf.ed.ac.uk/tadah/core) and
  [Tadah!MODELS](https://git.ecdf.ed.ac.uk/tadah/models) — fetched
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

## License

GPL-3.0; see [LICENSE](LICENSE).

## Reporting issues

Bugs and feature requests:
[Tadah!MLIP engine issues](https://git.ecdf.ed.ac.uk/tadah/mlip/-/issues).
User-facing questions belong on the umbrella project,
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip).
