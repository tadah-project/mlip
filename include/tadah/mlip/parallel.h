#pragma once
/* mlip::parallel_for -- a thin parallel-loop facade for the MLIP train/predict
 * kernels (Stage 2.3 of the pc_lammps fan-out work).
 *
 * Why this exists
 * ---------------
 * The design-matrix build and the predict loops used `#pragma omp parallel for`.
 * In ANNEAL_PT/PA every replica driver spawned its OWN OpenMP team for these, so
 * K replicas nested K teams and cores could not be shared across replicas: one
 * replica's train/predict could not use another replica's idle fan-out workers,
 * and vice-versa. Stage 2.2 already moved the pc_lammps fan-out onto one shared
 * std::thread pool (HPO's ScriptExecutor); this facade lets the same pool also
 * run the train/predict kernels, so a replica's training overlaps another's
 * fan-out and there is no nested OpenMP anywhere.
 *
 * How
 * ---
 *   - DEFAULT (no backend installed): parallel_for expands to the SAME
 *     `#pragma omp parallel for schedule(guided)` as before (or a serial loop
 *     without OpenMP). Standalone MLIP, the `tadah train|predict` CLI, and the
 *     MODELS/MLIP test suites are therefore byte-identical to pre-Stage-2.3.
 *   - BACKEND installed: the application layer (HPO) calls set_parallel_backend()
 *     at the start of each run (hpo_run) with a closure that submits the loop
 *     body to the run's ScriptExecutor pool, and clears it (an empty backend)
 *     when the run ends, however it ends. While it is installed, parallel_for
 *     routes every loop through that pool.
 *
 * MLIP cannot depend on HPO (HPO -> MLIP -> MODELS -> CORE), so the pool lives in
 * HPO and is injected here as a type-erased backend -- dependency-clean, and the
 * pool stays an application concern.
 *
 * Determinism
 * -----------
 * Every kernel that uses parallel_for writes DISJOINT output slots per index
 * (design-matrix row-blocks keyed by a precomputed `rows[s]`; per-structure
 * predictions/descriptors `stdb_(i)` / `st_desc_db(i)`) and performs no
 * cross-iteration reduction. The loop result is therefore independent of
 * execution order, OpenMP width, and backend: the facade only changes WHICH
 * thread runs an index and WHEN, never a value. (Same invariant that lets the
 * pc_lammps fan-out reorder freely.)
 *
 * Threading of the setter: install or clear the backend only while no
 * parallel_for runs (HPO: when a run starts and when it ends). It is not
 * meant to be swapped concurrently with kernels.
 */
#include <cstddef>
#include <exception>
#include <functional>
#include <vector>

namespace tadah {
namespace mlip {

/* A backend runs body(i) for every i in [0, n) -- in parallel if it wishes --
 * and BLOCKS until all indices have completed. */
using ParallelBackend =
    std::function<void(std::size_t n, const std::function<void(std::size_t)>& body)>;

/* Install (or clear, by passing an empty std::function) the process-wide
 * parallel backend. Call only while no parallel_for runs (HPO installs one
 * for each run and clears it when the run ends). */
void set_parallel_backend(ParallelBackend backend);

/* True iff a non-empty backend is installed. */
bool has_parallel_backend();

/* The installed backend. Precondition: has_parallel_backend(). Internal -- use
 * parallel_for, not this. */
const ParallelBackend& detail_parallel_backend();

/* Run body(i) for i in [0, n). Uses the installed pool backend if present, else
 * the historical `#pragma omp parallel for schedule(guided)` (serial without
 * OpenMP). body must be safe to run concurrently for distinct i (the MLIP
 * kernels satisfy this by writing disjoint slots -- see the determinism note).
 *
 * Exceptions: every index runs; if any body threw, an exception is rethrown
 * after the loop -- under OpenMP the one of the lowest index (an exception may
 * not leave an OpenMP region: it would call std::terminate), and the one the
 * backend reports otherwise (HPO's pool: the first to be thrown). */
template <class Body>
inline void parallel_for(std::size_t n, Body&& body) {
  if (has_parallel_backend()) {
    detail_parallel_backend()(n, [&](std::size_t i) { body(i); });
    return;
  }
#ifdef _OPENMP
  // One slot per index, written only by that index's thread: no lock.
  std::vector<std::exception_ptr> errors(n);
  #pragma omp parallel for schedule(guided)
  for (std::size_t i = 0; i < n; ++i) {
    try {
      body(i);
    } catch (...) {
      errors[i] = std::current_exception();
    }
  }
  for (const auto& e : errors)
    if (e) std::rethrow_exception(e);
#else
  for (std::size_t i = 0; i < n; ++i) body(i);
#endif
}

} // namespace mlip
} // namespace tadah
