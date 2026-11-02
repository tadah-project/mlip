#ifndef TADAH_MLIP_HPOTARGET_OPTIM_H
#define TADAH_MLIP_HPOTARGET_OPTIM_H

// The OPTIM lines of a target file read against a training configuration,
// as tadah hpo reads them: the refusals of a search target, the boxes of
// each selected value (an explicit box, or the AUTO box derived from the
// value's seed) and the refusals of a box. tadah hpo reads its OPTIM lines
// with OptimReader, and tadah analysis check_bounds its target file with
// read_optim_lines(), so both read every OPTIM line alike, with the same
// messages. The syntax of a line is hpotarget_syntax.h.

#include <cstddef>
#include <cstdint>
#include <istream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/mlip/hpotarget_syntax.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/models/descriptors/dm/rhoform.h>

namespace tadah {
namespace mlip {

/** Positive-only OPTIM keys (geometric AUTO boxes by default), from the
 *  refit hpotarget writer's key table (engines::hpotarget_group_specs()),
 *  so tadah hpo and the writer cannot disagree; another key is sign-free. */
bool optim_key_positive_only(const std::string& key);

/** \brief What the fit requires of a searched value. */
struct OptimFitValueRule {
  bool strict;      ///< > 0; else >= 0
  std::string why;  ///< the reason, in words
};

/** What the fit requires of a searched value of index @p index (1-based)
 *  of @p key (case-insensitive), which the low bound of its box must meet:
 *  a value > 0 (strict) for ESIGMA, the absolute parts FSIGMA (1) and
 *  SSIGMA (1) and every WDBFILE value (an AUTO box of them is geometric by
 *  default, RATIO 2), a value >= 0 for the relative parts FSIGMA (2) and
 *  SSIGMA (2) (an AUTO box of them is linear and clamped at 0) and for
 *  LAMBDA; nullopt for another key. */
std::optional<OptimFitValueRule> optim_fit_value_rule(const std::string& key,
                                                      std::size_t index);

/** OPTIM of a choice or a switch of the fit (ENORM ATOM|SQRT, ENERGY
 *  true|false) is refused: no continuous value. Returns the error message,
 *  "" for another key. */
std::string optim_switch_key_error(const std::string& key);

/** OPTIM of a retired key (the blueprint's retired state: EWEIGHT,
 *  FWEIGHT, SWEIGHT, WDBFILE_AUTO, EWEIGHT_TEMP, UNCERTAINTY, WCOV, WNOISE,
 *  ERROR) is refused with why it was retired, what replaces it and its
 *  conversion (core::retired_key_text). Returns the error message, "" for
 *  another key. */
std::string optim_retired_key_error(const std::string& key);

/** OPTIM WEIGHTS, the coefficients each evaluation fits (a searched value
 *  would be replaced by the fit; OPTIM FIXWEIGHT searches the coefficients
 *  FIXINDEX holds fixed), and OPTIM EMBF0, a choice of the DM_REAM
 *  embedding reference, are refused; compared case-insensitively. Returns
 *  the error message, "" for another key. */
std::string optim_fitted_key_error(const std::string& key);

/** OPTIM on structure data is refused: the density form (RHOFORM), the
 *  descriptor types (TYPE2B, TYPEMB), the cutoff classes (RCTYPE2B,
 *  RCTYPEMB), the knot-degree lists (KDEG2B, KDEGMB, KDEGEMB), the
 *  rigid-dimer layout (DIMER), the pinned-weight indices (FIXINDEX) and the
 *  model (MODEL); compared case-insensitively. Returns the error message
 *  for such a key, "" for a searchable one. (OPTIM EMBF0 has its own
 *  message, optim_fitted_key_error.) */
std::string optim_structure_key_error(const std::string& key);

/** A discrete value of a key whose other values are continuous: the first
 *  OALGO value (the training algorithm; the second, the rcond M, is
 *  searchable). @p index is the OPTIM index (1-based among the key's
 *  numeric values). Returns the error message, "" for a searchable index. */
std::string optim_discrete_index_error(const std::string& key,
                                       std::size_t index);

/** OPTIM on an internal key (the blueprint's INTERNAL_KEYS: DSIZE,
 *  RCUT*MAX, ESTDEV, ...) is refused: tadah computes those values and no
 *  configuration sets them. Returns the error message, "" for another key. */
std::string optim_internal_key_error(const std::string& key);

/** OPTIM on a key applied once, when the run starts and reads its data
 *  (LSCALE, ESHIFT, ESHIFT_ATOM, ESHIFT_DBATOM, EFILTER, FFILTER,
 *  ZERO_COM_FORCE) is refused, whatever the model: no evaluation would take
 *  a candidate value of it. Compared case-insensitively. Returns the error
 *  message, "" for another key. */
std::string optim_startup_only_key_error(const std::string& key);

/** OPTIM on a legacy many-body key (CGRIDMB, SGRIDMB, AMPGRIDMB, CEMBFUNC,
 *  SEMBFUNC, RCUTMB) that @p ctx does not store, while its many-body term
 *  is DM_Embed components (DESCRIPTOR blocks): refused naming the
 *  component path of each block that holds the parameter. A stored key (a
 *  DM_REAM term beside the blocks) is searchable. Returns the error
 *  message, "" otherwise. */
std::string optim_component_key_error(const core::Context& ctx,
                                      const std::string& key);

/** OPTIM on a value that the key's formats (blueprint value_type, every
 *  alternative) take only as an integer -- SBASIS, the first value of
 *  OALGO -- is refused: a searched value is continuous. @p config holds
 *  the key's tokens, @p pos the token of OPTIM index @p index. Returns the
 *  error message, "" for a value the key also takes as a fraction. */
std::string optim_integer_index_error(const std::string& key,
                                      const std::vector<std::string>& config,
                                      std::size_t pos, std::size_t index);

/** \brief What the OPTIM lines read besides the training configuration. */
struct OptimSeed {
  /** The values of a potential that its descriptors do not read, left out
   *  of the configuration (models::D_mJoin::expand_grids): an OPTIM index
   *  past the values kept names them. */
  std::vector<models::D_mJoin::LeftOut> left_out;
  /** The density form: RHOFORM of a DM_REAM configuration, or the FORM of
   *  the DENSITY of its one DM_Embed component; nullopt otherwise. */
  std::optional<models::RhoForm> rho_form;
};

/** \brief Prepares the training configuration @p ctx for its OPTIM lines,
 *  as tadah hpo prepares it: its DESCRIPTOR blocks resolved and its radial
 *  grids expanded (models::D_mJoin::expand_grids, which leaves out the
 *  values of a potential its descriptors do not read), and its density
 *  form read (a DM_REAM configuration without RHOFORM is refused with the
 *  migration text, which @p command names). Call once per configuration. */
OptimSeed prepare_optim_seed(core::Context& ctx, const std::string& command);

/** \brief How the box of a value was obtained. */
enum class OptimBoxOrigin : std::uint8_t {
  Explicit,         ///< the line's bounds
  AutoLinear,       ///< AUTO: seed +- max(FRAC |seed|, FLOOR, GROUP_FLOOR max|seed|)
  AutoGeometric,    ///< AUTO: [seed/RATIO, seed*RATIO]
  AutoLinearNonneg  ///< AUTO linear, its low bound clamped at 0
};

/** \brief The box of one value an OPTIM line selects. */
struct OptimBox {
  std::size_t index = 0;  ///< the index as the line writes it (FIXWEIGHT: the coefficient)
  std::size_t slot = 0;   ///< 0-based position among the key's tokens
  double lo = 0.0, hi = 0.0;
  OptimBoxOrigin origin = OptimBoxOrigin::Explicit;
};

/** \brief An OPTIM line read against the training configuration. */
struct OptimLine {
  std::string key;          ///< upper-cased
  bool named = false;       ///< a DM_Embed component path (holds a '.')
  std::size_t line = 0;     ///< the last physical line of the line
  /** The key's tokens in the training configuration (a path: its values,
   *  written exactly). */
  std::vector<std::string> config;
  OptimShape shape = OptimShape::Linear;
  double c = 1.0;           ///< the c of ASINH <c>
  bool tagged = false;      ///< the line writes LIN, LOG or ASINH <c>
  bool auto_box = false;
  std::vector<OptimBox> boxes;  ///< in the order of the index list
};

/** \brief The OPTIM lines of one target file, read in order, as tadah hpo
 *  reads them. One reader reads the lines of one file: a value two lines
 *  select is refused at the second. */
class OptimReader {
 public:
  /** @p ctx: the training configuration, prepared (prepare_optim_seed),
   *  which a key of a line inserts its blueprint default into, as tadah hpo
   *  inserts it; @p seed: what prepare_optim_seed returned. */
  OptimReader(core::Context& ctx, OptimSeed seed);
  /** The OPTIM line whose text after the word OPTIM is @p rest and whose
   *  last physical line is @p line. Throws std::runtime_error (or
   *  std::logic_error) with tadah hpo's text, without the line's frame. */
  OptimLine read(std::istringstream& rest, std::size_t line);

 private:
  core::Context& context;
  OptimSeed seed_;
  /** Per key, the line that selects each token (0: none). */
  std::map<std::string, std::vector<std::size_t>> selected_;
};

/** \brief An OPTIM line of a target file and what it reads. */
struct OptimTargetLine {
  TargetLine source;
  OptimLine optim;
};

/** \brief Every logical line of @p in whose key is OPTIM (blocks are not
 *  followed: a line inside one is read too), read in order with one
 *  OptimReader on @p ctx and @p seed (prepare_optim_seed). Throws
 *  TargetLineError at the first line refused, the refusal of an unfinished
 *  last line (kTargetUnfinishedLineMsg) included. */
std::vector<OptimTargetLine> read_optim_lines(std::istream& in,
                                              core::Context& ctx,
                                              const OptimSeed& seed);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_HPOTARGET_OPTIM_H
