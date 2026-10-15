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
#include <tadah/models/descriptors/dm/rhoform.h>

namespace tadah {
namespace mlip {
namespace engines {

/** \brief The OPTIM key groups an hpotarget carries, with their bound rule.
 *  positive_only keys (the width channels) get the lo > 0 clamp. The
 *  density-form rule (AMPGRIDMB lo >= 0 under RHOFORM NONNEG/MONOTONE,
 *  models::optim_key_lower_bound) is applied on top of this static table,
 *  which `tadah hpo` shares. */
struct OptimGroupSpec {
  const char *key;
  bool positive_only;
};

/** \brief The 8 group keys in emission order (shape blocks + FIXWEIGHT). */
const std::array<OptimGroupSpec, 8> &hpotarget_group_specs();

/** \brief The group bound floor: max(1e-8, 1e-3·max|v|), rounded to 6
 *  significant digits. The rounding makes the floor reproducible from the
 *  PRINTED bounds (%.8e, 9 significant digits; --retarget recovers the
 *  group scale to ~1e-9 relative), so floor-dominated bounds are
 *  byte-stable under repeated retargeting instead of jittering in their
 *  last digit. */
double optim_group_floor(double gmax);

/** \brief Decimal grid for OPTIM bounds: 10^(floor(log10(|centre| +
 *  |half-width|)) - 6), at most 7 significant digits of a bound, coarser
 *  than the print resolution (bounds are printed %.8e, 9 significant
 *  digits). Centre and half-width are snapped to this grid before
 *  emission, so the printed bounds are EXACT decimals and --retarget
 *  recovers the centre exactly — the round-trip is a fixed point instead
 *  of jittering in the last digit (the naive midpoint of two printed
 *  bounds carries the LARGER bound's rounding error into the finer print
 *  grid of the smaller one). */
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
                             bool positive_only,
                             tadah::models::OptimLowerBound lower =
                                 tadah::models::OptimLowerBound::Free);

/** \brief One `OPTIM KEY (i) lo hi` line per parameter; bounds are
 *  v +- max(pct%·|v|, floor) with floor = max(1e-8, 1e-3·max|v| of the
 *  group) so near-zero parameters keep a searchable window.
 *  @p lower NonNegative (AMPGRIDMB under RHOFORM NONNEG/MONOTONE) clamps
 *  lo = max(0, v - d), so an amplitude of 0 gets [0, d]. */
void emit_optim_group(std::ostream &os, const std::string &key,
                      const tadah::core::v_type &vals, double pct,
                      bool positive_only, bool commented = false,
                      tadah::models::OptimLowerBound lower =
                          tadah::models::OptimLowerBound::Free);

/** \brief The parsed hpotarget header line that records a positive density
 *  form: "# RHOFORM <form>: AMPGRIDMB bounds are non-negative (lo = max(0,
 *  v - d))". --retarget reads the form from it (the hpotarget carries no
 *  potential) to keep those bounds non-negative. Only for NONNEG and
 *  MONOTONE. */
std::string rhoform_bounds_line(tadah::models::RhoForm form);

/** \brief The parsed comment a starter target records for the knot centres
 *  of a DM_Embed density with CUTOFF TYPE NONE, whose knots end within the
 *  fixed RADIUS: "# <ID>.DENSITY.CENTRES knot-support cap: RADIUS = <R>
 *  (...)". The boxes of those centres end at R (on the bound grid, at most
 *  R), and --retarget, which sees only the file, reads the cap from it. */
std::string knot_support_cap_line(const std::string &path, double radius);

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
 *  The named parameters of each DM_Embed component (COMPONENT_PARAMETER
 *  lines and their commented OPTIM groups) follow the pair section under a
 *  header of their own, "# ---- <ID> component (DM_Embed) density and
 *  embedding shape ----"; a zero-referenced embedding is noted as
 *  REFERENCE ZERO (a component's key) or EMBF0 ZERO (a DM_REAM term's).
 *  RCUT2B/RCUTMB are read from the pot; for KNOT channels the commented
 *  `# OPTIM RCUT*` scalar is replaced by an explanatory comment (the
 *  effective cutoff is the largest knot — vary CGRID* instead). A pot with
 *  a DM_REAM block must declare RHOFORM (DensityFormError otherwise); under
 *  NONNEG/MONOTONE the header carries rhoform_bounds_line() and the
 *  AMPGRIDMB bounds are non-negative. */
void write_hpotarget(const std::string &path, const tadah::core::Context &pot,
                     double pct);

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_HPOTARGET_WRITER_H
