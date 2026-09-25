# Changelog

All notable changes to **Tadah.MLIP** (the MLIP module — high-level training / predict / data-balance orchestration) are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> Note: the umbrella project that ships the `tadah` binary is also called **Tadah.MLIP**; its changelog lives at the umbrella repo root and aggregates across all modules. This file is the per-module log for the `MLIP/` library layer.

## [Unreleased]

## [1.6.0] — 2026-09-25

### Added
- Per-dataset energy and force RMSE derivatives (used by per-dataset HPO validation targets).
- `mlip::parallel_for` facade for the train/predict kernels; defaults to the previous OpenMP loop, and lets `tadah hpo` route the work onto its shared worker pool.

### Changed
- **Potentials of DM_REAM models always carry `EMBF0`** (`get_param_file` of M_BLR/M_KRR and `dumpPotFile` normalise the key: ZERO when absent, canonical spelling; a stray key is removed from potentials without a DM_REAM block).
- `tadah predict` (and the descriptor/pairwise analysis engines): a potential and a task configuration that set different `EMBF0` values are refused; a task value applies to a potential without the key (INFO).
- `tadah train`: a HINT when a DM_REAM model under `EMBF0 ZERO` is trained without BIAS and without ESHIFT (the model then has no free per-atom constant).
- `tadah refit`: reads `EMBF0` from the refit configuration (default ZERO); the self-test, curve and compare dumps use the zero-referenced table under ZERO and the log reports the removed table F(0); `REFIT_CONSTRAIN_F0 false` under ZERO warns (no effect). `refit --edit` logs the pot's mode (WARNING when the input has no `EMBF0` line). The hpotarget header notes the ESHIFT/BIAS requirement under ZERO.
- **Dataset files follow NUMERIC as significant digits, default 18 = exact.** `tadah data convert|merge|split|sample|dedup|balance` (`StructureDB::dump_to_file`) wrote NUMERIC fixed decimals: values below ~0.01 lost significant digits and values below 5e-19 became 0, so a rewritten dataset trained on slightly different targets and weights than the command computed (e.g. a `balance --rescale` eweight). Every number is now written with `%.{N}g` in `N+7` columns and reads back bit-identically; an explicit NUMERIC below 17 is honoured with a WARNING (`StructureDB::dataset_digits`). Files usually get shorter. The LAMMPS data and VASP POSCAR exports of `tadah data write` follow the same rule, masses included (`StructureDB::export_digits`); the CASTEP cell export keeps NUMERIC decimal places. The default precision of `dump_to_file` is 18 (was 12 decimals) and of `StructureWriter` 18 (was 10). Test: `test_dataset_precision.cpp`.
- **Pre-flight audit: no "critical hyperparameter RCUT2B/RCUTMB is fixed" warning for bump potentials under a hard cutoff.** `run_critical_optim_audit` already skipped knot channels (RCUT is derived from the largest knot); it now also skips `D2_Blip`/`D2_Blip5` pair channels and `DM_REAM` bump densities whose cutoff is discontinuous (`Cut_Dummy`, the refit default): there RCUT is the fitted truncation, every bump support ends at or before it, a larger value changes no prediction and a smaller one only trips the bump-support gate, and `tadah hpo` warns when `OPTIM RCUT*` is written anyway. Under a smooth cutoff the warning stays (the damping tail depends on RCUT). Test: `test_preflight_audit.cpp`.
- **hpotarget writer/`--retarget`: bump-channel OPTIM boxes are no longer capped.** `tadah hpo` now enforces the bump-support invariant `c + w/eta <= RCUT` per candidate (`BUMPGATE CANDIDATE`, its new default — violating combinations are rejected before training), so the writer emits CGRID/SGRID bounds at full symmetric width and `--retarget` stops re-applying the cap (`apply_bump_cap`/`BumpCapResult` removed). The parsed contract comment (`# RCUT2B bump-support contract: w = ...`) is kept — reworded — and still drives the seed-spill warning (a pot whose own bumps spill is rejected by hpo's startup SEED check) and the `--set` infeasible-corner warning; the ACTIVE-`OPTIM RCUT*` wasted-dimension note is softened to the new semantics (the sampled cutoff only reshapes the feasible region). Old capped files retarget fine: the centre-recovery rule for asymmetric (hi-capped) lines is unchanged, and retarget(fresh, same pct) remains byte-idempotent.
- Framework version `1.6.0`, read from CORE's `blueprint/blueprint_tadah.toml` after the Tadah dependencies are fetched, so the module reports the right version in every layout (sibling checkout or FetchContent). A missing blueprint is now a configure error instead of a silent `0.0.0`.
- Tests take Catch2 and its test discovery from cmake-tools (`tadah_setup_tests()`, `include(Catch)`); the vendored `tests/catch2/catch.hpp`, `tests/Catch.cmake` and `tests/CatchAddTests.cmake` are removed. Every discovered CTest entry now selects exactly its own test case and passes `--warn NoTests`, so an entry that matches no test case fails instead of passing.
- cmake-tools is fetched at `v1-dev` instead of `main`.
- The pre-flight audit hint for a critical hyperparameter that is fixed in the seed and absent from every `OPTIM` block names the cause: the seed value "misses the neighbour distances of the training data (e.g. a cutoff or radial grid that does not overlap the neighbour shells)". It said "(BASINS-style)", after an experimental HPO strategy that the user documentation never described and that is removed in 1.6.0.
- The documentation of `classify_dataset` states its decision rules instead of pointing to a BASINS investigation note that is not in the repository.
- The two `NNFinder: real CH4 phase-I structure` tests are reported by CTest as Skipped, not Passed, when their optional database is absent (`TADAH_TEST_PHASEI_DB` unset and no `ch4_phasei_S_U67.tadahdb` in the working directory): they call `TADAH_SKIP` (cmake-tools' TADAH-SKIP marker) instead of returning after a warning. A database that `TADAH_TEST_PHASEI_DB` names and that cannot be opened fails them.
- **A `DIMER` potential whose descriptors name specific element pairs is refused when it is loaded** (`tadah train`, `tadah predict`, and every other user of `DescriptorsCalc`). `calc_dimer` evaluates every atom as Z=1, whatever its element (with the pair weight of Z=1, not the `WATOM` weight of the element), so a `TYPE2B`/`TYPEMB` declared for `N N` applied to no pair it sees and the model predicted from zero descriptors without a word (`tadah predict` gave 0 eV for the published Tadah!N2 potential with `N N` pairs; -0.00184 eV/atom with its `* *`). Every `DescriptorsCalc` constructor now calls `models::require_dimer_element_pairs()` (MODELS 1.6.0): `DIMER evaluates every atom as Z=1; declare TYPE2B/TYPEMB with * * element pairs`. Test: `test_descriptors_calc_dimer.cpp`.
- The README badges and links and the pointers to the umbrella's community files (`CITATION.md`, `CONTRIBUTING.md`, …) point to the `v1-dev` line: GitLab `v1-dev`, `https://tadah.readthedocs.io/v1-dev/`. They pointed to `develop`.

### Removed
- The legacy MPI training build (`TADAH_BUILD_MPI`), documented as not functional: `MPI_Trainer`, `TrainerHost`, `TrainerWorker` and their ScaLAPACK/BLACS declarations (`trainer.h`), the host/worker paths of `TrainEngine` with their MPI-only rejections of `LSCALE`, `ESHIFT`, `EWEIGHT_TEMP`, `WDBFILE_AUTO` and `--uncertainty`, the rank-0 guards of the engines, `Scheduler::distributeTasks` and the per-task barrier, `Task::serializeBinary` and `Task::deserializeBinary` (only `distributeTasks` called them), and the MPI link. It never linked ScaLAPACK outside HPO, its `Logger` differed from CORE's, and its solver ignored `LAMBDA`, `NORM`, `BIAS` and KRR. Training runs in one process, with OpenMP threads.

### Fixed
- `Scheduler::execute` returns the first non-zero engine status (and stops) instead of discarding it, so a failed task no longer exits with success.
- The test `write_optim_block: scalar -> (1) ; vector -> (1-N)` never ran under CTest: the vendored Catch discovery script split its name at the `;` into two entries, which matched no test case and still passed. It now runs and passes.
- **`DIMER` models: every interatomic distance was zero.** `DescriptorsCalc::calc_dimer` assigned each displacement to the copy that `Mat6R3C::row()` returns, so the six distances of a molecule pair stayed zero and every configuration got the same descriptors, in training and in `tadah predict`. The displacements are written with `Mat6R3C::set_row()`, and `row()` returns a const copy (CORE 1.6.0), so the assignment that lost them no longer compiles. `pair_style tadah` computes its own distances and was not affected; `tadah predict` with the published `Tadah!N2` potential now gives the energies LAMMPS gives. Test: `test_descriptors_calc_dimer.cpp`.

## [1.5.1] — 2026-09-25

### Fixed
- **Potential files are written losslessly.** `TrainEngine::dumpPotFile` (train, `refit`, `refit --edit`) writes through `Context::write_potential`: every non-integer number with NUMERIC significant digits, default 18, so a potential reloaded from pot.tadah is bit-identical to the one in memory. NUMERIC is never written into the file; an explicit NUMERIC below 17 is honoured with a WARNING. `tadah refit` now honours NUMERIC from its configuration (it was ignored). `refit --edit` also honours a NUMERIC carried by the edited potential -- every 1.5.0 `tadah hpo` potential has `NUMERIC 12`; delete that line -- with a WARNING naming its source, and no longer promises an identical prediction when NUMERIC < 17. Tests: `test_dump_pot_file.cpp`, `test_refit_engine.cpp`, `test_refit_edit_mode.cpp`.
- **LSCALE / ESHIFT are stamped into pot.tadah exactly.** They had 6 significant digits, so predict and HPO applied a slightly different transform than training; newly trained potentials may differ from older ones by up to ~1e-5 eV/atom. Test: `test_load_transforms.cpp`.
- Readers accept subnormal values (`core::to_double`): FIXWEIGHT, the energy-only line of a dataset structure, `refit --retarget` bounds, hpotarget contracts, VASP POSCAR coordinates and CASTEP cell numbers. A FIXWEIGHT written as `1e-310` failed with "FIXWEIGHT entry is not a number", and a POSCAR exported by `tadah data write` with a subnormal coordinate failed to load. Tests: `test_weight_provisioning.cpp`, `test_dataset_precision.cpp`.
- **`Structure::save` writes the z force, and every double exactly.** The z force column held `force(1)` (the y component); weights and energy had 6 significant digits, and cell, stress, positions and forces 6 decimals. Every double now has 17 significant digits, so `save()` -> `read()` round-trips bit-exactly; `save(path)` throws when the file cannot be opened or written, the caller's stream format is restored, and a one-letter symbol no longer writes a NUL byte. `Structure::save` has no CLI caller (public API only). Test: `test_structure.cpp`.

## [1.4.0] — 2026-07-14

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
