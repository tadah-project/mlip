[![pipeline](https://git.ecdf.ed.ac.uk/tadah/mlip/badges/v1-dev/pipeline.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/commits/v1-dev)
[![coverage](https://git.ecdf.ed.ac.uk/tadah/mlip/badges/v1-dev/coverage.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/commits/v1-dev)
[![license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![version](https://img.shields.io/badge/version-1.6.0-blue.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/tags)

# MLIP (engine library)

Engine library of the [Tadah!](https://tadah.readthedocs.io/v1-dev/) MLIP
toolchain. Implements the engines that back every `tadah` subcommand:

- **Training** — design-matrix construction, regression dispatch,
  potential-file emission.
- **Prediction** — apply a trained `pot.tadah` to datasets or
  individual structures; per-frame analytics.
- **Data engines** — dataset conversion from external sources
  (CASTEP / VASP / AFLOW / COD / CIF), de-duplication, merging,
  splitting, sampling, energy shifts per element.
- **Analysis engines** — cutoff-function plots, descriptor
  calculation.
- **Properties engine** — pairwise energy curves.
- **Load-time data transforms** — `LSCALE`, `ESHIFT` /
  `ESHIFT_ATOM` / `ESHIFT_DBATOM`, `EFILTER`, `FFILTER`,
  `ZERO_COM_FORCE`, equilibrium-volume diagnostic.
- **Error scales of a fit** — the weight of every row of a design
  matrix from `ESIGMA`, `FSIGMA`, `SSIGMA`, `ENORM` and `ENERGY` and
  the priorities of the data (`WDBFILE` and each structure's
  `eweight`, `fweight` and `sweight`), its observations and its
  effective parameters (`row_weights.h`).

## You don't need to use this directly

MLIP (the engine library) is an internal module of the Tadah!
MLIP toolchain. It is fetched automatically (via CMake `FetchContent`)
by the umbrella project
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip), which links
this library and provides the `tadah` CLI binary plus documentation.
End users install the umbrella project — see the
[installation guide](https://tadah.readthedocs.io/v1-dev/installation.html).

This repository is broken out separately so the hyperparameter
optimisation engine [HPO](https://git.ecdf.ed.ac.uk/tadah/hpo)
can link the same training/prediction code path that the CLI does.

## Contributing

Build instructions, dependencies, and layout live in
[CONTRIBUTING.md](CONTRIBUTING.md). The full multi-repo contributor
guide is on the umbrella project.

## License

GPL-3.0; see [LICENSE](LICENSE).

## Reporting issues

Bugs and feature requests:
[MLIP engine issues](https://git.ecdf.ed.ac.uk/tadah/mlip/-/issues).
For end-user questions please use the umbrella project
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip).
