#ifndef TADAH_MLIP_ENGINES_HPOTARGET_WRITER_H
#define TADAH_MLIP_ENGINES_HPOTARGET_WRITER_H

// The `tadah refit` hpotarget emitter, exported so it is reusable outside
// RefitEngine::run: the --edit mode regenerates a target from an edited pot
// Context, and the --retarget rewriter reuses the exact bound/format rules
// (emit_optim_group's floor + clamp + precision) so a rewrite at the same
// percentage reproduces a fresh emission byte-for-byte.

#include <array>
#include <iosfwd>
#include <string>

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>

namespace tadah {
namespace mlip {
namespace engines {

/** \brief The OPTIM key groups an hpotarget carries, with their bound rule.
 *  positive_only keys (the width channels) get the lo > 0 clamp. */
struct OptimGroupSpec {
  const char *key;
  bool positive_only;
};

/** \brief The 8 group keys in emission order (shape blocks + FIXWEIGHT). */
const std::array<OptimGroupSpec, 8> &hpotarget_group_specs();

/** \brief The group bound floor: max(1e-8, 1e-3·max|v|), rounded to 6
 *  significant digits. The rounding makes the floor reproducible from the
 *  8-significant-digit PRINTED bounds (--retarget recovers the group scale
 *  to ~1e-9 relative), so floor-dominated bounds are byte-stable under
 *  repeated retargeting instead of jittering in their last digit. */
double optim_group_floor(double gmax);

/** \brief Decimal grid for OPTIM bounds: 10x the print resolution of the
 *  larger bound (bounds are printed with 8 significant digits). Centre and
 *  half-width are snapped to this grid before emission, so the printed
 *  bounds are EXACT decimals and --retarget recovers the centre exactly —
 *  the round-trip is a fixed point instead of jittering in the last digit
 *  (the naive midpoint of two 8-digit bounds carries the LARGER bound's
 *  rounding error into the finer print grid of the smaller one). */
double optim_bound_quantum(double centre, double halfwidth);

/** \brief Snap x to the decimal grid q (round-half-away like the printer). */
double quantize_optim(double x, double q);

/** \brief Directional snaps: down = toward -inf, up = toward +inf, so a
 *  derived bound satisfies its inequality EXACTLY on the decimal grid. */
double quantize_optim_down(double x, double q);
double quantize_optim_up(double x, double q);

/** \brief One quantized `OPTIM` bound: the emission rule of
 *  emit_optim_group for a single value, exposed so the paired (CGRID,
 *  SGRID) emitter and --retarget share the exact numerics. */
struct OptimBound {
  double lo = 0.0, hi = 0.0;
  double v = 0.0;  ///< the grid-snapped centre the bounds are built around
  double q = 0.0;  ///< the decimal grid the bounds live on
};
OptimBound optim_group_bound(double val, double pct, double floor,
                             bool positive_only);

/** \brief One `OPTIM KEY (i) lo hi` line per parameter; bounds are
 *  v +- max(pct%·|v|, floor) with floor = max(1e-8, 1e-3·max|v| of the
 *  group) so near-zero parameters keep a searchable window. */
void emit_optim_group(std::ostream &os, const std::string &key,
                      const tadah::core::v_type &vals, double pct,
                      bool positive_only, bool commented = false);

/** \brief Scalar variant used for the RCUT2B/RCUTMB lines. */
void emit_optim_scalar(std::ostream &os, const std::string &key, double v,
                       double pct, double lo_clamp, bool commented = false);

/** \brief Paired (CGRID*, SGRID*) emission for a BUMP channel under a hard
 *  cutoff: identical bytes to two emit_optim_group calls (bounds are
 *  emitted at full width — `tadah hpo`'s per-candidate bump-support gate
 *  rejects infeasible combinations at evaluation time). Warns (once per
 *  channel, naming the indices) when the pot's own bumps spill — hpo's
 *  startup SEED check rejects such a file until the pot is repaired. */
void emit_optim_group_pair(std::ostream &os, const std::string &ckey,
                           const tadah::core::v_type &cvals,
                           const std::string &skey,
                           const tadah::core::v_type &svals, double pct,
                           bool commented, double w, double rcut);

/** \brief Write a full hpotarget starter file from a refit pot Context.
 *  RCUT2B/RCUTMB are read from the pot; for KNOT channels the commented
 *  `# OPTIM RCUT*` scalar is replaced by an explanatory comment (the
 *  effective cutoff is the largest knot — vary CGRID* instead). */
void write_hpotarget(const std::string &path, const tadah::core::Context &pot,
                     double pct);

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_HPOTARGET_WRITER_H
