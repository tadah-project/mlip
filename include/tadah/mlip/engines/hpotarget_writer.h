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

/** \brief One `OPTIM KEY (i) lo hi` line per parameter; bounds are
 *  v +- max(pct%·|v|, floor) with floor = max(1e-8, 1e-3·max|v| of the
 *  group) so near-zero parameters keep a searchable window. */
void emit_optim_group(std::ostream &os, const std::string &key,
                      const tadah::core::v_type &vals, double pct,
                      bool positive_only, bool commented = false);

/** \brief Scalar variant used for the RCUT2B/RCUTMB lines. */
void emit_optim_scalar(std::ostream &os, const std::string &key, double v,
                       double pct, double lo_clamp, bool commented = false);

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
