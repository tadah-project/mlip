# Changelog

All notable changes to **Tadah.MLIP** (the MLIP module — high-level training / predict / data-balance orchestration) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> Note: the umbrella project that ships the `tadah` binary is also called **Tadah.MLIP**; its changelog lives at the umbrella repo root and aggregates across all modules. This file is the per-module log for the `MLIP/` library layer.

## [Unreleased]

### Added
- **`--retarget --rcut2b <r> / --rcutmb <r>`**: declare a bump channel's fixed cutoff in the same run — the file's `OPTIM RCUT*` reference line is recentred on `r`, stray ACTIVE `OPTIM RCUT*` lines are commented out with a warning (searching RCUT on a gated bump channel is a wasted dimension), and the rewritten CGRID/SGRID boxes are truncated to `c + w/eta <= r`. Replaces the two-step `--set 'RCUT2B(1)=lo,hi'` + `--pct` dance after manually raising a pot's cutoff.

### Changed
- **hpotarget writer: bump channels under a hard (Cut_Dummy) cutoff get gate-compliant OPTIM boxes.** Per blip, the emitted CGRID upper / SGRID lower bounds are capped (directionally quantized, so the cap is exact on the bound grid) to the bump-support invariant `c_hi + w/eta_lo <= RCUT` that `tadah hpo` now hard-errors on, and a parsed contract comment (`# RCUT2B bump-support contract: w = ...`) records the half-width for `--retarget`. A pot whose own bump spills is emitted uncapped with a warning (the repair tool `refit --edit remove` also writes an hpotarget, so the writer never throws); knot/Gaussian/FIXWEIGHT emissions are byte-identical to before.
- **`--retarget` understands the bump-support contract**: recovers the centre of a hi-capped CGRID line from its LOWER bound (mirror of the existing lo-clamped SGRID repair), re-applies the cap after every rewrite (fresh(pct) -> retarget(pct) stays byte-identical; any retargeted file is byte-stable), and lets `--set` win with a warning when it violates the contract corner.
- `--retarget`: with duplicate or ACTIVE `OPTIM RCUT*` lines the bump-support cap now binds against the smallest reachable cutoff (active line -> its rewritten LOW bound, commented -> centre) and warns, instead of trusting whichever line comes first in the file.
- `refit --edit --add` passes the pot's cutoff kind to `potedit::add_basis`: a bump support crossing rcut is a hard error for Cut_Dummy pots (with the minimal `--eta` in the message), a warning for smooth cutoffs.

## [1.3.0-beta.1] — 2026-05-20

The MLIP layer gains in-place HPO support, load-time dataset transforms, and a wave of correctness fixes.

### Added
- `Trainer::reinit()` — rebuilds the trainer for new descriptor parameters in place, without freeing/reconstructing the full object. Enables HPO to mutate descriptor cutoffs / parameters between evaluations cheaply.
- Load-time data transforms applied during dataset load before training: `LSCALE`, `ESHIFT` (per-species), `ESHIFT_ATOM` (derive from isolated-atom configs), `ESHIFT_DBATOM`, `EWEIGHT_TEMP` (Boltzmann-style energy weighting), `EFILTER` / `FFILTER` (drop configs outside thresholds), `WDBFILE` / `WDBFILE_AUTO` (per-database weights), `ZERO_COM_FORCE`. Applied uniformly to `tadah train`, `tadah predict`, `tadah hpo`, and `tadah data balance`; persisted into `pot.tadah` for prediction round-trip.
- mJoin support in basis-function plotting.
- Augmented design matrix logged alongside the raw DM.
- `TADAH_ENABLE_ASAN` CMake option.
- **HPO bound-validation pipeline** (Phases 2–5): design-matrix `ColumnBlock` / `ColumnLayout` types (`design_matrix/column_block.h`); `DC_Base::layout()` + `DescriptorsCalc<>::layout()` override; dataset stats (`data_stats_compute.{h,cpp}` — `compute_data_stats` + `classify_dataset`, with the NN-sweep `sample_size` parameter wired in from blueprint `AUDIT_SAMPLE`); per-`ParamMeta` data-driven OPTIM bound suggestion (`bound_suggester.{h,cpp}`); pre-flight audit (`preflight_audit.{h,cpp}`, `AuditMode::{Off,Warn,Error}`). Catches the BASINS class of silent training failures (descriptors that collapse to zero weights when bounds put them outside the dataset's support).

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
- Test build routed through `cmake-tools/cmake/TadahTesting.cmake` helpers (`tadah_declare_testing` / `tadah_add_tests`).
- README: dropped a dead link to the removed engineer-facing `HPO/LOAD_TRANSFORMS.md` note (content now lives in `hpo.rst`).

## [1.2.0] and earlier (condensed)

The `v1.0.0` → `v1.1.1` releases were the "internal playground" era. MLIP grew as the high-level training / predict orchestration around the regressor and trainer code, with build-system polish, removal of the unused per-module `nlopt` dependency, and a sequence of small bug fixes. No published changelog covers this era; treat anything before `1.3.0-beta.1` as undocumented in the modern sense.
