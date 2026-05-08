#ifndef TADAH_MLIP_CHECK_BOUNDS_H
#define TADAH_MLIP_CHECK_BOUNDS_H

// Phase 7 — `tadah analysis check_bounds`. CI-friendly dry-run.
//
// Reads an HPOTARGET file's `OPTIM <KEY> (range) <lo> <hi>` lines,
// builds a bound box, samples corners (full 2^k for k ≤ 12, otherwise
// `min(2^k, ⌈k·log₂(k)⌉)` Latin-hypercube samples), and runs the
// Phase 5 pre-flight audit at each corner. Returns 1 if ANY corner
// produces a FAIL finding, 0 otherwise. Intended for CI gating.

#include <cstdint>
#include <string>
#include <vector>

#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

class StructureDB;

/** \brief One bound entry from the HPOTARGET file. Each entry covers
 *  one (key, arg_pos) tuple with explicit `(lo, hi)`. Vector entries
 *  expanded — `OPTIM CGRID2B (1-4) 0 5` becomes 4 separate Bound
 *  rows with arg_pos 0..3. */
struct Bound {
  std::string key{};
  int         arg_pos{-1};   // -1 scalar; 0..n-1 vector
  double      lo{0.0};
  double      hi{0.0};
};

/** \brief Parse an HPOTARGET file for `OPTIM <KEY> (range) <lo> <hi>`
 *  rows. Tolerant — unrecognised lines are skipped. Throws on any
 *  malformed `OPTIM` line (e.g. missing range or bounds). */
std::vector<Bound> parse_hpo_target_bounds(const std::string& path);

/** \brief Sample corner combinations of `bounds`. For `bounds.size() <= 12`
 *  returns the full `2^k` corner set. For larger problems returns a
 *  Latin-hypercube subsample of `min(2^k, ⌈k·log₂(k)⌉)` rows. Each
 *  inner vector has the same length as `bounds` and contains either
 *  `bounds[i].lo` or `bounds[i].hi`.
 */
std::vector<std::vector<double>>
sample_corners(const std::vector<Bound>& bounds, std::uint64_t seed = 0);

/** \brief Run the Phase 5 audit at every corner of the bound box.
 *
 *  Side-effect: emits one INFO line per corner via Logger and a
 *  WARNING summarising the worst corner.
 *
 *  \return number of corners that produced any FAIL finding.
 */
struct CheckBoundsResult {
  std::size_t n_corners{0};
  std::size_t n_corners_failing{0};
  std::vector<std::size_t> failing_corner_indices{};
};

CheckBoundsResult check_bounds_at_corners(
    StructureDB&                     stdb,
    tadah::core::Context             seed,
    const std::vector<Bound>&        bounds,
    const std::vector<std::vector<double>>& corners);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_CHECK_BOUNDS_H
