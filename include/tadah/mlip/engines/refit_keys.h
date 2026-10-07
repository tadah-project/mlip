#ifndef TADAH_MLIP_ENGINES_REFIT_KEYS_H
#define TADAH_MLIP_ENGINES_REFIT_KEYS_H

// The keys of a `tadah refit` configuration: its REFIT_MODE and the mode
// rule (which keys each mode reads), and its numbers, read strictly with
// their ranges: the one reader of RefitEngine (validate(), run()) and of the
// HPO library's compact fitter (HPORefitEngine::compact_fit).

#include <tadah/core/context.h>
#include <tadah/mlip/key_provenance.h>

#include <array>
#include <optional>
#include <string>

namespace tadah {
namespace mlip {
namespace engines {

/** The fit `tadah refit` runs (REFIT_MODE). */
enum class RefitMode {
  Compact,  ///< the compact fitter (the HPO library, with Ceres): the default
  Dense     ///< the dense windowed least squares (models::EAMRefit::fit)
};

/** REFIT_MODE of @p c in any letter case (as REFIT_OBJECTIVE is read):
 *  'compact', the default without the key, or 'dense'. Another value throws
 *  std::runtime_error naming it as written:
 *    "tadah refit: REFIT_MODE must be 'compact' (the default) or 'dense',
 *     got 'sparse'" */
RefitMode read_refit_mode(const tadah::core::Context &c);

/** REFIT_OBJECTIVE of @p c in any letter case: true for 'curve', false for
 *  'energy', the default without the key (no blueprint default is
 *  inserted). Another value throws std::runtime_error:
 *    "tadah refit: REFIT_OBJECTIVE must be 'energy' (default) or 'curve';
 *     got 'shape'" (the value in lower case). */
bool read_refit_curve_objective(const tadah::core::Context &c);

/** The refusal of a top-level OPTIMIZER block, the grammar before the
 *  stage blocks, which neither REFIT_MODE reads: the migration text of
 *  check_refit_mode_keys (in either mode) and of the compact fitter's
 *  parser (HPO's RefitParser::parse):
 *    "tadah refit: a top-level OPTIMIZER block is no longer supported. The
 *     stage pipeline is now INIT -> EXPLORE -> REFINE -> FUSE, ... The
 *     Refitting page of the documentation has examples (Stage blocks)." */
std::string top_level_optimizer_refusal();

/** The mode rule: refuse a key of @p c that the configuration's REFIT_MODE,
 *  @p mode, does not read. Each key's mode is stated once, here (the table
 *  in refit_keys.cpp):
 *   - read by REFIT_MODE dense only (the dense fit and its basis
 *     escalation): REFIT_LEVEL, N2B, NRHO_BASIS, NF_BASIS,
 *     REFIT_MAX_ESCALATIONS, REFIT_PLACEMENT, REFIT_CONSTRAIN_F0;
 *   - read by REFIT_MODE compact only (the compact fitter and its stage
 *     pipeline): every REFIT_COMPACT_* key, REFIT_CURVE_WEIGHTING,
 *     REFIT_CURVE_RELAX, REFIT_CURVE_DERIV_WEIGHT, the INIT, EXPLORE,
 *     REFINE and FUSE blocks, and REFIT_OBJECTIVE curve (REFIT_OBJECTIVE
 *     energy passes in both);
 *   - read by REFIT_OBJECTIVE curve only, in compact mode:
 *     REFIT_CURVE_WEIGHTING. (REFIT_CURVE_RELAX and
 *     REFIT_CURVE_DERIV_WEIGHT have no effect under energy, and pass there:
 *     their values are checked in either objective.)
 *   - read by neither mode: a top-level OPTIMIZER block, refused first, in
 *     either mode, with top_level_optimizer_refusal(), to which REFIT_MODE
 *     dense adds " Under REFIT_MODE dense, which reads no stage block,
 *     remove the OPTIMIZER block."
 *  The other keys are read in both modes. A key counts when @p c stores it
 *  (Context::has_stored) and @p provenance, when given, does not record it
 *  as a blueprint default that the task factory added (KeyOrigin::Default:
 *  REFIT_COMPACT_MERGE, REFIT_CURVE_*, REFIT_OBJECTIVE and
 *  REFIT_CONSTRAIN_F0 have one). The first key refused throws
 *  std::runtime_error naming it, the mode in effect ("(the default)"
 *  without a REFIT_MODE key) and the mode that reads it:
 *    "tadah refit: N2B has no effect under REFIT_MODE compact (the
 *     default): only REFIT_MODE dense reads it; remove N2B, or set
 *     REFIT_MODE dense"
 *    "tadah refit: the REFINE block has no effect under REFIT_MODE dense:
 *     only REFIT_MODE compact reads it; remove the REFINE block, or set
 *     REFIT_MODE compact"
 *    "tadah refit: REFIT_CURVE_WEIGHTING has no effect under
 *     REFIT_OBJECTIVE energy (the default): only REFIT_OBJECTIVE curve
 *     reads it; remove REFIT_CURVE_WEIGHTING, or set REFIT_OBJECTIVE curve"
 *  REFIT_OBJECTIVE is read with read_refit_curve_objective (in either
 *  mode). */
void check_refit_mode_keys(const tadah::core::Context &c, RefitMode mode,
                           const KeyProvenance *provenance);

/** The refit configuration numbers. A key the configuration does not store
 *  (Context::has_stored: no blueprint default is inserted) has the default
 *  below, the refit's own; the optional ones are left empty, their default
 *  being another's (models::EAMRefitConfig's for the reachability window,
 *  REFIT_LEVEL's for N2B/NRHO_BASIS/NF_BASIS, the basis family's for
 *  REFIT_COMPACT_STALL/INSERT_TOPK). The mode each key belongs to is the
 *  mode rule's (check_refit_mode_keys); every number is range-checked
 *  whatever the mode. */
struct RefitNumbers {
  // both modes
  double tol_ambient = 0.1;        ///< REFIT_TOL_AMBIENT, meV/atom, > 0
  double tol_compressed = 1.0;     ///< REFIT_TOL_COMPRESSED, meV/atom, > 0
  // (default: models::EAMRefitConfig's)
  std::optional<double> reach_smin;    ///< REFIT_REACH_SMIN, 0.1 to 1
  std::optional<double> reach_margin;  ///< REFIT_REACH_MARGIN, > 0
  std::optional<double> r_lo_frac;     ///< REFIT_R_LO_FRAC, in (0, 1)
  double output_grid_scale = 1.0;  ///< OUTPUT_GRID_SCALE, > 0
  double optim_pct = 5.0;          ///< OPTIM_PCT, > 0
  // REFIT_MODE dense
  std::optional<int> n2b, nrho, nF;  ///< N2B, NRHO_BASIS, NF_BASIS: 1..1024
  int max_escalations = 4;           ///< REFIT_MAX_ESCALATIONS, 0..INT_MAX
  // REFIT_MODE compact
  std::array<int, 3> nmax = {40, 40, 40};  ///< REFIT_COMPACT_NMAX [phi, rho,
                                           ///< F], one or three values, each
                                           ///< 1..INT_MAX, the largest's
                                           ///< 4 x NMAX sample grid an int
                                           ///< (NMAX <= 536870911)
  int samples = 800;               ///< REFIT_COMPACT_SAMPLES, 1..INT_MAX
  int opt_every = 1;               ///< REFIT_COMPACT_OPT_EVERY, 1..INT_MAX
  std::optional<int> stall;        ///< REFIT_COMPACT_STALL, 1..INT_MAX
  std::optional<int> insert_topk;  ///< REFIT_COMPACT_INSERT_TOPK, 1..INT_MAX
  int jitters = 2;                 ///< REFIT_COMPACT_JITTERS, 0..INT_MAX - 1
                                   ///< (its jitters + 1 candidates an int)
  unsigned seed = 12345;           ///< REFIT_COMPACT_SEED, 0..4294967295
  double share = 0.5;              ///< REFIT_COMPACT_SHARE, > 0
  double curve_deriv_weight = 0.0; ///< REFIT_CURVE_DERIV_WEIGHT, >= 0
  double curve_relax = 0.05;       ///< REFIT_CURVE_RELAX, in (0, 1], in
                                   ///< every REFIT_OBJECTIVE
};

/** Read the refit configuration numbers of @p c. A real number takes the
 *  number grammar of the hpotarget file (core::parse_decimal_normal: no hex,
 *  inf, nan or subnormal), an integer decimal digits only, and each must lie
 *  in its range (RefitNumbers; the largest REFIT_COMPACT_NMAX's curve sample
 *  grid of 4 x NMAX points must fit an int). A config file must also pass
 *  CORE's blueprint check of the key's type, which refuses '.5' in every
 *  number and a sign ('+3') in an integer. Otherwise throws
 *  std::runtime_error naming the key and the token as written:
 *    "tadah refit: REFIT_TOL_AMBIENT must be a positive number, got '0'"
 *    "tadah refit: REFIT_COMPACT_NMAX must be an integer from 1 to
 *     2147483647, got '+3'"
 *  followed by " (too small: ...)" for a number below the smallest normal
 *  double. */
RefitNumbers read_refit_numbers(const tadah::core::Context &c);

/** OPTIM_PCT alone, read as read_refit_numbers() reads it (> 0, default 5):
 *  the number of the --edit and --retarget modes. */
double read_optim_pct(const tadah::core::Context &c);

}  // namespace engines
}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_ENGINES_REFIT_KEYS_H
