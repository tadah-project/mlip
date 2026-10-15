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

/** The REFIT_BASIS spellings tadah refit fits, one family each, as
 *  models::basis_kind_from_name, the one table, which the fit reads too:
 *   - the one-sided knot families: Knot5, KNOT5, knot5, Knot3, KNOT3,
 *     knot3, KnotMix, KNOTMIX, knotmix, Mixed, MIXED;
 *   - Blip: Blip, BLIP, blip;
 *   - Blip5: Blip5, BLIP5, blip5, M6;
 *   - Gaussian (REFIT_MODE dense only, experimental: RefitEngine warns):
 *     Gaussian, gaussian, BP, D2_BP.
 *  Another spelling names no family: check_refit_basis refuses it. */
bool refit_basis_is_knot_family(const std::string &basis);
bool refit_basis_is_blip(const std::string &basis);      ///< see above
bool refit_basis_is_blip5(const std::string &basis);     ///< see above
bool refit_basis_is_gaussian(const std::string &basis);  ///< see above

/** Refuse a REFIT_BASIS of @p c that names no family (above), in either
 *  REFIT_MODE, before the EAM file is read (RefitEngine::validate, run()):
 *    "tadah refit: REFIT_BASIS 'Blipp' is no basis family of tadah refit:
 *     Blip (Blip, BLIP, blip), Blip5 (...), the knot families Knot5, Knot3
 *     and KnotMix (...), or, under REFIT_MODE dense only, the experimental
 *     Gaussian (Gaussian, gaussian, BP, D2_BP)"
 *  It was fitted as Blip (models::basis_kind_from_string was lenient) while
 *  RefitEngine warned of the Gaussian basis. */
void check_refit_basis(const tadah::core::Context &c);

/** REFIT_LEVEL of @p c in any letter case: 'coarse', 'balanced' (the
 *  default without the key), 'fine' or 'accurate', in lower case. Another
 *  value throws std::runtime_error naming the four and the value as
 *  written:
 *    "tadah refit: REFIT_LEVEL must be 'coarse', 'balanced' (the default),
 *     'fine' or 'accurate' (in any letter case), got 'acurate'"
 *  (any other value fitted the balanced preset). */
std::string read_refit_level(const tadah::core::Context &c);

/** REFIT_PLACEMENT of @p c in any letter case: 'window' (the default
 *  without the key) or 'uniform', in lower case. Another value throws:
 *    "tadah refit: REFIT_PLACEMENT must be 'window' (the default) or
 *     'uniform' (in any letter case), got 'unifrom'"
 *  (any other value ran the window placement). */
std::string read_refit_placement(const tadah::core::Context &c);

/** Whether the configuration sets @p key: @p c stores it, and @p provenance,
 *  when given, does not record it as a blueprint default the task factory
 *  added (KeyOrigin::Default). The rule of every refit key check, and of the
 *  output names of --edit and --retarget (an explicit -o pot.tadah is the
 *  user's, the stored default pot.tadah is not). */
bool refit_key_configured(const tadah::core::Context &c,
                          const KeyProvenance *provenance,
                          const std::string &key);

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
 *     REFIT_MAX_ESCALATIONS, REFIT_PLACEMENT, REFIT_CONSTRAIN_F0; and
 *     REFIT_CONSTRAIN_F0, the F(0) pin of an EMBF0 FREE fit, only under
 *     EMBF0 FREE (refused under EMBF0 ZERO, written or the default:
 *     "tadah refit: REFIT_CONSTRAIN_F0 has no effect under EMBF0 ZERO (the
 *     default): only EMBF0 FREE reads it; remove REFIT_CONSTRAIN_F0, or set
 *     EMBF0 FREE");
 *   - read by REFIT_MODE compact only (the compact fitter and its stage
 *     pipeline): every REFIT_COMPACT_* key, REFIT_CURVE_WEIGHTING,
 *     REFIT_CURVE_RELAX, REFIT_CURVE_DERIV_WEIGHT, the INIT, EXPLORE,
 *     REFINE and FUSE blocks, and REFIT_OBJECTIVE curve (REFIT_OBJECTIVE
 *     energy passes in both); and REFIT_COMPONENT_ITERATIONS, the
 *     iteration limit of the compact fit of DM_Embed components, only with
 *     DESCRIPTOR blocks ("tadah refit: REFIT_COMPONENT_ITERATIONS has no
 *     effect without DESCRIPTOR blocks: ...");
 *   - read by the knot families only, in compact mode (the compact fitter
 *     applies them to the one-sided knot kinds): REFIT_COMPACT_MERGE,
 *     REFIT_COMPACT_OPT_EVERY and REFIT_COMPACT_INSERT_TOPK, refused when
 *     REFIT_BASIS is a Blip family (refit_basis_is_blip, _blip5);
 *   - read by REFIT_OBJECTIVE curve only, in compact mode: every curve
 *     knob, REFIT_CURVE_WEIGHTING, REFIT_CURVE_RELAX and
 *     REFIT_CURVE_DERIV_WEIGHT; and REFIT_CURVE_RELAX, the floor of the
 *     curve weighting, only with REFIT_CURVE_WEIGHTING true (its default);
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
 *    "tadah refit: REFIT_COMPACT_MERGE has no effect under REFIT_BASIS
 *     Blip5: only the knot families (REFIT_BASIS Knot5, the compact
 *     default, Knot3 or KnotMix) read it; remove REFIT_COMPACT_MERGE, or
 *     set REFIT_BASIS Knot5" (REFIT_BASIS as written)
 *    "tadah refit: REFIT_CURVE_WEIGHTING has no effect under
 *     REFIT_OBJECTIVE energy (the default): only REFIT_OBJECTIVE curve
 *     reads it; remove REFIT_CURVE_WEIGHTING, or set REFIT_OBJECTIVE curve"
 *    "tadah refit: REFIT_CURVE_RELAX has no effect under
 *     REFIT_CURVE_WEIGHTING false: only REFIT_CURVE_WEIGHTING true (the
 *     default) reads it; remove REFIT_CURVE_RELAX, or set
 *     REFIT_CURVE_WEIGHTING true"
 *  REFIT_OBJECTIVE is read with read_refit_curve_objective (in either
 *  mode). */
void check_refit_mode_keys(const tadah::core::Context &c, RefitMode mode,
                           const KeyProvenance *provenance);

/** The modes of `tadah refit` on existing refit outputs, which read no EAM
 *  file. */
enum class RefitOutputMode {
  Edit,     ///< --edit <pot.tadah>
  Retarget  ///< --retarget <file.hpotarget>
};

/** The rule of --edit and --retarget: refuse a key of @p c that a refit of
 *  an EAM file reads and @p mode does not (the same table as the mode
 *  rule's, refit_keys.cpp):
 *   - the keys of either REFIT_MODE (REFIT_MODE, REFIT_BASIS,
 *     REFIT_TOL_AMBIENT, REFIT_TOL_COMPRESSED, REFIT_REACH_SMIN,
 *     REFIT_REACH_MARGIN, REFIT_R_LO_FRAC, REFIT_OBJECTIVE,
 *     OUTPUT_GRID_SCALE, SETFL_OUT, REFIT_LOG), of REFIT_MODE dense only
 *     and of REFIT_MODE compact only (check_refit_mode_keys), the stage
 *     blocks among them, and a top-level OPTIMIZER block, which no mode
 *     reads;
 *   - under --retarget also HPOFILEOUT, which a refit and --edit read (the
 *     hpotarget they write).
 *  NUMERIC, the precision of the files a command writes, is no key of the
 *  refit and is not refused: --edit reads it for the potential it writes,
 *  and --retarget writes only an hpotarget, whose bounds are printed %.8e
 *  whatever NUMERIC is (a task file's NUMERIC above its first TASK line
 *  applies to every task). (EAM, and RHOFORM and EMBF0 under --retarget,
 *  have their own refusals: validate_refit_mode_keys, refit_modes.h.) A
 *  key counts as in check_refit_mode_keys (stored and not a blueprint
 *  default). The first key refused throws std::runtime_error naming it
 *  and what reads it:
 *    "tadah refit --edit: N2B has no effect under --edit: only a refit of
 *     an EAM file in REFIT_MODE dense reads it; remove N2B"
 *    "tadah refit --retarget: the REFINE block has no effect under
 *     --retarget: only a refit of an EAM file in REFIT_MODE compact reads
 *     it; remove the REFINE block"
 *    "tadah refit --edit: REFIT_TOL_AMBIENT has no effect under --edit:
 *     only a refit of an EAM file reads it; remove REFIT_TOL_AMBIENT"
 *    "tadah refit --edit: the OPTIMIZER block has no effect under --edit:
 *     no mode of tadah refit reads a top-level OPTIMIZER block; remove the
 *     OPTIMIZER block"
 *    "tadah refit --retarget: HPOFILEOUT has no effect under --retarget:
 *     only a refit of an EAM file and --edit read it; remove HPOFILEOUT
 *     (the file --retarget writes is -o, OUTFILE)" */
void check_refit_output_mode_keys(const tadah::core::Context &c,
                                  RefitOutputMode mode,
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
  double curve_relax = 0.05;       ///< REFIT_CURVE_RELAX, in (0, 1]
  // REFIT_MODE compact, DESCRIPTOR blocks (the component compact fit)
  int component_iterations = 60;   ///< REFIT_COMPONENT_ITERATIONS, 1..INT_MAX
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
