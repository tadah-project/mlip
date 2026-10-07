#ifndef TADAH_MLIP_ENGINES_REFIT_KEYS_H
#define TADAH_MLIP_ENGINES_REFIT_KEYS_H

// The numbers of a `tadah refit` configuration, read strictly with their
// ranges: the one reader of RefitEngine (the dense path, validate()) and of
// the HPO library's compact fitter (HPORefitEngine::compact_fit).

#include <tadah/core/context.h>

#include <array>
#include <optional>

namespace tadah {
namespace mlip {
namespace engines {

/** The refit configuration numbers. A key the configuration does not store
 *  (Context::has_stored: no blueprint default is inserted) has the default
 *  below, the refit's own; the optional ones are left empty, their default
 *  being another's (models::EAMRefitConfig's for the reachability window,
 *  REFIT_LEVEL's for N2B/NRHO_BASIS/NF_BASIS, the basis family's for
 *  REFIT_COMPACT_STALL/INSERT_TOPK). */
struct RefitNumbers {
  // both modes
  double tol_ambient = 0.1;        ///< REFIT_TOL_AMBIENT, meV/atom, > 0
  double tol_compressed = 1.0;     ///< REFIT_TOL_COMPRESSED, meV/atom, > 0
  // (default: models::EAMRefitConfig's)
  std::optional<double> reach_smin;    ///< REFIT_REACH_SMIN, in (0, 1]
  std::optional<double> reach_margin;  ///< REFIT_REACH_MARGIN, > 0
  std::optional<double> r_lo_frac;     ///< REFIT_R_LO_FRAC, in (0, 1)
  double output_grid_scale = 1.0;  ///< OUTPUT_GRID_SCALE, > 0
  double optim_pct = 5.0;          ///< OPTIM_PCT, > 0
  // dense mode
  std::optional<int> n2b, nrho, nF;  ///< N2B, NRHO_BASIS, NF_BASIS: 1..1024
  int max_escalations = 4;           ///< REFIT_MAX_ESCALATIONS, 0..INT_MAX
  // compact mode
  std::array<int, 3> nmax = {40, 40, 40};  ///< REFIT_COMPACT_NMAX [phi, rho,
                                           ///< F], one or three values, each
                                           ///< 1..INT_MAX
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
 *  in its range (RefitNumbers). A config file
 *  must also pass CORE's blueprint check of the key's type, which refuses,
 *  for instance, '.5' and '+3'. Otherwise throws std::runtime_error naming
 *  the key and the token as written:
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
