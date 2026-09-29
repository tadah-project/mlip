#ifndef TADAH_MLIP_ENGINES_HPOTARGET_RETARGET_H
#define TADAH_MLIP_ENGINES_HPOTARGET_RETARGET_H

// `tadah refit --retarget`: rewrite the OPTIM search bounds of an existing
// hpotarget file. Line-based by design — HPOTargets' parser strips comments
// and is lossy, so the only faithful transformation preserves every
// non-OPTIM line byte-for-byte and rewrites OPTIM lines (active AND
// commented, keeping their comment state) around the recovered centres.
//
// Centre recovery: the emitter writes symmetric bounds v ± max(pct·|v|,
// floor), so v is the midpoint — except where the positive-only lo-clamp
// fired (width channels, RCUT scalars, and AMPGRIDMB when the file header
// declares RHOFORM NONNEG/MONOTONE); those are repaired by inverting the
// unclamped UPPER bound using the emission percentage of the line's group:
// its `# KEY  (+-X%` comment (which a per-key --pct rewrites), else the file
// header. An RCUT scalar line has no such comment from the writer: a
// per-key --pct RCUT2B=N adds one above the line ("# RCUT2B  (+-N%, 1
// params)") and removes it when the line's percentage is the global one
// again, so every percentage is a fixed point. Exactness is at the printed
// precision (8 significant digits).
//
// Density form: the file never sees the potential, so the writer records a
// positive RHOFORM in a parsed header line (rhoform_bounds_line); --rhoform
// adds that line to a file written without it. With it, AMPGRIDMB bounds
// are rewritten with lo = max(0, v - d), a line at that declared clamp
// (lo = 0) is recovered without a warning, and a --set with lo < 0 on
// AMPGRIDMB is refused.

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <tadah/models/descriptors/dm/rhoform.h>

namespace tadah {
namespace mlip {
namespace engines {

struct RetargetOptions {
  double pct = 5.0;                        ///< global half-width, percent
  std::map<std::string, double> pct_per_key;  ///< --pct KEY=N overrides
  struct SetOverride {
    std::string key;
    std::size_t index = 0;  ///< 1-based, as written in the OPTIM line
    double lo = 0.0, hi = 0.0;
  };
  std::vector<SetOverride> sets;           ///< --set 'KEY(i)=lo,hi' (last)
  /** --rcut2b / --rcutmb: declare a bump channel's FIXED cutoff in one step.
   *  Keyed by channel suffix ("2B" / "MB"). The file's OPTIM RCUT{sfx}
   *  reference line(s) are recentred on the value (ACTIVE ones are commented
   *  out with a warning: with the bump-support gate, searching RCUT on a
   *  bump channel is a wasted dimension) and the bump-support cap binds
   *  against it. Ignored with a warning on knot channels (RCUT is derived
   *  from the largest knot there). */
  std::map<std::string, double> rcut_override;
  /** --rhoform: the density form of the potential the file searches
   *  around. Under NONNEG/MONOTONE a file without a RHOFORM header gets one
   *  (rhoform_bounds_line, where the writer puts it) and its AMPGRIDMB boxes
   *  are rewritten non-negative; SIGNED adds nothing. A header that names
   *  another form (any header, under SIGNED) is an error naming its line. */
  std::optional<tadah::models::RhoForm> rhoform;
};

struct RetargetReport {
  std::size_t n_rewritten = 0;
  std::size_t n_active = 0;
  std::size_t n_commented = 0;
  std::vector<std::string> warnings;
  /** The density form of the file's RHOFORM header line (NONNEG or
   *  MONOTONE; empty when the file has none). */
  std::string rhoform;
};

/** \brief Rewrite the OPTIM bounds in `lines` in place (the unit-testable
 *  core). Non-OPTIM lines are untouched; group `(+-X%, N params...)` header
 *  comments and the file-header percentage are updated to the new values
 *  (an RCUT line's comment added or removed, see above). Throws on a
 *  per-key percentage for a key without an OPTIM line, on malformed --set
 *  overrides (unknown KEY(i), a bound that is not finite, hi <= lo, or a lo
 *  the file's density form refuses: models::optim_low_bound_rule, e.g.
 *  AMPGRIDMB lo < 0 under a positive RHOFORM), and on a RHOFORM header line
 *  whose form is not a density form (the error names the line). */
RetargetReport retarget_hpotarget_lines(std::vector<std::string> &lines,
                                        const RetargetOptions &opts);

/** \brief File wrapper: read `in`, rewrite, write to `out` (or atomically
 *  back onto `in` via temp + rename when in_place). Throws when out == in
 *  without in_place, or on I/O failure. */
RetargetReport retarget_hpotarget_file(const std::string &in,
                                       const std::string &out, bool in_place,
                                       const RetargetOptions &opts);

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_HPOTARGET_RETARGET_H
