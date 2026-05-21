[![pipeline](https://git.ecdf.ed.ac.uk/tadah/mlip/badges/develop/pipeline.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/commits/develop)
[![coverage](https://git.ecdf.ed.ac.uk/tadah/mlip/badges/develop/coverage.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/commits/develop)
[![license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![version](https://img.shields.io/badge/version-1.3.0--beta.1-blue.svg)](https://git.ecdf.ed.ac.uk/tadah/mlip/-/tags)

# Tadah!MLIP (engine library)

Engine library of the [Tadah!](https://tadah.readthedocs.io/) MLIP
toolchain. Implements the engines that back every `tadah` subcommand:

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

## You don't need to use this directly

Tadah!MLIP (the engine library) is an internal module of the Tadah!
MLIP toolchain. It is fetched automatically (via CMake `FetchContent`)
by the umbrella project
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip), which links
this library and provides the `tadah` CLI binary plus documentation.
End users install the umbrella project — see the
[installation guide](https://tadah.readthedocs.io/en/latest/installation.html).

This repository is broken out separately so the hyperparameter
optimisation engine [Tadah!HPO](https://git.ecdf.ed.ac.uk/tadah/hpo)
can link the same training/prediction code path that the CLI does.

## Contributing

Build instructions, dependencies, and layout live in
[CONTRIBUTING.md](CONTRIBUTING.md). The full multi-repo contributor
guide is on the umbrella project.

## License

GPL-3.0; see [LICENSE](LICENSE).

## Reporting issues

Bugs and feature requests:
[Tadah!MLIP engine issues](https://git.ecdf.ed.ac.uk/tadah/mlip/-/issues).
For end-user questions please use the umbrella project
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip).
