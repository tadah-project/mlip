# Changelog

All notable changes to **Tadah.MLIP** (the MLIP module — high-level training / predict / data-balance orchestration) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> Note: the umbrella project that ships the `tadah` binary is also called **Tadah.MLIP**; its changelog lives at the umbrella repo root and aggregates across all modules. This file is the per-module log for the `MLIP/` library layer.

## [Unreleased]

Nothing yet.

## [1.3.0-beta.1] — TBD

The MLIP layer gains in-place HPO support, load-time dataset transforms, and a wave of correctness fixes.

### Added
- `Trainer::reinit()` — rebuilds the trainer for new descriptor parameters in place, without freeing/reconstructing the full object. Enables HPO to mutate descriptor cutoffs / parameters between evaluations cheaply.
- Load-time data transforms applied during dataset load before training: `LSCALE`, `ESHIFT` (per-species), `ESHIFT_ATOM` (derive from isolated-atom configs), `ESHIFT_DBATOM`, `EWEIGHT_TEMP` (Boltzmann-style energy weighting), `EFILTER` / `FFILTER` (drop configs outside thresholds), `WDBFILE` / `WDBFILE_AUTO` (per-database weights), `ZERO_COM_FORCE`. Applied uniformly to `tadah train`, `tadah predict`, `tadah hpo`, and `tadah data balance`; persisted into `pot.tadah` for prediction round-trip.
- mJoin support in basis-function plotting.
- Augmented design matrix logged alongside the raw DM.
- `TADAH_ENABLE_ASAN` CMake option.

### Changed
- Improved terminal output (status lines and progress reporting).

### Fixed
- **MPI / BLACS lifetime** — `MPI_Type_free` and `blacs_gridexit` moved into the destructor; `TrainerHost::run()` now resets state and clears `wpckgs` before rebuild. Fixes leaks and double-init that previously prevented HPO from reusing a TrainerHost across evaluations.
- Memory leak in the pairwise predict engine (2b and mb model / fb pointers were not being deleted).
- `value_N` index now clamped — out-of-bounds when `default_value` was longer than the configured value count no longer crashes.
- `data_balance_engine` threshold handling and `predict_engine` cleanup.
- `bspan` formula and shift computation in `calc_binned` — corrects binned-descriptor evaluation.
- R² seg-fault when the metric was computed on forces.
- `--config` now parses without spurious errors.
- Running with `-F` and no `--outfile` no longer errors out.
- Test fixtures no longer hardcode `/home/marcin` paths (`TADAH_TEST_PHASEI_DB` env var override added).

### Internal
- Substantive README rewrite (Session B).
- Community / governance pointer files added.
- CI/CD: lint + release stages, VERSION wired from blueprint TOML.

## [1.2.0] and earlier (condensed)

The `v1.0.0` → `v1.1.1` releases were the "internal playground" era. MLIP grew as the high-level training / predict orchestration around the regressor and trainer code, with build-system polish, removal of the unused per-module `nlopt` dependency, and a sequence of small bug fixes. No published changelog covers this era; treat anything before `1.3.0-beta.1` as undocumented in the modern sense.
