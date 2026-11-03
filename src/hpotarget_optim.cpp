#include <tadah/mlip/hpotarget_optim.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/blueprint_validator.h>
#include <tadah/core/number_format.h>
#include <tadah/core/retired_keys.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <tadah/models/descriptors/dm/embf0.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tadah {
namespace mlip {

namespace {

/* The 0-based position among @p tokens of the ui-th numeric token
 * (core::is_number), tokens.size() when there are fewer: an OPTIM index
 * counts the numeric values of its key. */
std::size_t optim_value_position(const std::vector<std::string>& tokens,
                                 std::size_t ui) {
  std::size_t offset = 0;
  std::size_t count = 0;
  for (const auto& t : tokens) {
    if (core::is_number(t)) count++;
    if (count == ui) return offset;
    offset++;
  }
  return offset;
}

/* Whether @p value is finite, by its bits (neither -ffast-math nor a
 * floating-point mode can defeat the test). */
bool value_is_finite(double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<double>::is_iec559);
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

}  // namespace

/* Positive-only OPTIM keys, from the same table the refit hpotarget writer
 * uses (MLIP hpotarget_writer.h) so HPO and the writer cannot disagree;
 * keys unknown to that table are sign-free. */
bool optim_key_positive_only(const std::string& key) {
  for (const auto& g : engines::hpotarget_group_specs())
    if (key == g.key) return g.positive_only;
  return false;
}

std::string optim_fitted_key_error(const std::string& key) {
  const std::string k = core::to_upper(key);
  if (k == "WEIGHTS")
    return "OPTIM WEIGHTS: WEIGHTS holds the coefficients each evaluation "
           "fits, so a searched value would be replaced by the fit; FIXINDEX "
           "and FIXWEIGHT in the training configuration hold coefficients "
           "fixed, and OPTIM FIXWEIGHT (<coefficient>) <low> <high> searches "
           "them";
  if (k == "EMBF0")
    return "OPTIM EMBF0: EMBF0 (ZERO or FREE) chooses the embedding "
           "reference of a DM_REAM many-body term, a choice and not a number "
           "to search; the training configuration sets it";
  return "";
}

std::string optim_structure_key_error(const std::string& key) {
  static const struct { const char* key; const char* what; } structure[] = {
    {"RHOFORM",  "RHOFORM (SIGNED|NONNEG|MONOTONE) selects the DM_REAM density "
                 "form, a discrete property of the potential (its amplitudes "
                 "were fitted in that form)"},
    {"TYPE2B",   "TYPE2B names the two-body descriptor and its structure "
                 "tokens (basis code, sizes, species)"},
    {"TYPEMB",   "TYPEMB names the many-body descriptor and its structure "
                 "tokens (basis code, sizes, species)"},
    {"RCTYPE2B", "RCTYPE2B names the two-body cutoff function"},
    {"RCTYPEMB", "RCTYPEMB names the many-body cutoff function"},
    {"KDEG2B",   "KDEG2B lists the two-body knot degrees (3|5)"},
    {"KDEGMB",   "KDEGMB lists the density knot degrees (3|5)"},
    {"KDEGEMB",  "KDEGEMB lists the embedding knot degrees (3|5)"},
    {"DIMER",    "DIMER (on/off, the bond length, whether the intramolecular "
                 "distance enters the descriptors) declares the rigid-dimer "
                 "layout of the training data (its bond length is that of "
                 "the data's molecules)"},
    {"FIXINDEX", "FIXINDEX names the coefficients FIXWEIGHT holds fixed "
                 "(OPTIM FIXWEIGHT searches their values)"},
    {"MODEL",    "MODEL names the model and its basis function or kernel "
                 "(and an integer option)"},
  };
  const std::string k = core::to_upper(key);
  for (const auto& e : structure)
    if (k == e.key)
      return "OPTIM " + k + " is not a valid search target: " + e.what +
             ", structure data rather than a continuous parameter; the "
             "training configuration sets it.";
  return "";
}

std::string optim_discrete_index_error(const std::string& key,
                                       std::size_t index) {
  if (core::to_upper(key) == "OALGO" && index == 1)
    return "OPTIM OALGO (1) is not a valid search target: the first OALGO "
           "value selects the training algorithm (1 GELSD, 2 GELS, 3 SVD, "
           "4 Cholesky), a discrete choice rather than a continuous "
           "parameter; the training configuration sets it. The second value, "
           "the rcond M of options 1 and 3, is continuous: OPTIM OALGO (2) "
           "<low> <high>.";
  return "";
}

std::string optim_component_key_error(const core::Context& ctx,
                                      const std::string& key) {
  static const struct { const char* key; bool density; const char* field; }
      legacy[] = {{"CGRIDMB", true, "CENTRES"},   {"SGRIDMB", true, ""},
                  {"AMPGRIDMB", true, "AMPLITUDES"},
                  {"CEMBFUNC", false, "CENTRES"}, {"SEMBFUNC", false, ""},
                  {"RCUTMB", true, "CUTOFF.RADIUS"}};
  const std::string k = core::to_upper(key);
  const auto* row = std::find_if(std::begin(legacy), std::end(legacy),
                                 [&](const auto& r) { return k == r.key; });
  if (row == std::end(legacy) || ctx.has_stored(k)) return "";
  const auto specs = models::embed_specs(ctx);
  if (specs.empty()) return "";
  std::string paths;
  for (const auto& s : specs) {
    const auto& c = row->density ? s.density : s.embedding;
    std::string field = row->field;
    if (field.empty())  // a width: the coordinate the block declares
      field = c.basis == models::EmbedBasis::Knot5
                  ? ""
                  : c.coordinates == models::EmbedCoordinates::Widths ? "WIDTHS"
                                                                      : "ETA";
    const std::string id = core::to_upper(s.id);
    const std::string channel = row->density ? ".DENSITY." : ".EMBEDDING.";
    paths += (paths.empty() ? "" : ", ") +
             (field.empty()
                  ? id + channel + "* (a Knot5 component has no width)"
                  : id + channel + field +
                        (field == "CUTOFF.RADIUS" && s.cutoff == "NONE"
                             ? " (a parameter when the CUTOFF TYPE is not "
                               "NONE)"
                             : ""));
  }
  return "OPTIM " + k + ": the potential has no " + k + " key: its many-body "
         "term is DM_Embed component(s) (DESCRIPTOR blocks), whose " +
         (row->density ? "density" : "embedding") + " parameter is the named "
         "path " + paths + "; an OPTIM line names the path, OPTIM <path> "
         "(indices) <low> <high> (the starter target file of tadah refit "
         "names the paths; a target file written for a DM_REAM many-body "
         "term names " + k + ")";
}

std::string optim_internal_key_error(const std::string& key) {
  const std::string k = core::to_upper(key);
  if (core::BlueprintParser::getInternalKeysMapRef().count(k) == 0) return "";
  return "OPTIM " + k + " is not a valid search target: " + k +
         " is an internal key, a value tadah computes itself (no "
         "configuration sets it, so no evaluation can take a candidate "
         "value of it).";
}

std::string optim_startup_only_key_error(const std::string& key) {
  static const char* const once[] = {
      "LSCALE",  "ESHIFT",  "ESHIFT_ATOM",   "ESHIFT_DBATOM",
      "EFILTER", "FFILTER", "ZERO_COM_FORCE"};
  const std::string k = core::to_upper(key);
  for (const char* o : once)
    if (k == o)
      return "OPTIM " + k + " is not a valid search target: " + k +
             " prepares the data and is applied once, when the run starts "
             "and reads them, so no evaluation would take a candidate value "
             "of it: its value is the training configuration's.";
  return "";
}

std::optional<OptimFitValueRule> optim_fit_value_rule(
    const std::string& key, std::size_t index) {
  const std::string k = core::to_upper(key);
  if (k == "LAMBDA")
    return OptimFitValueRule{false,
                        "LAMBDA is the ridge penalty, a number >= 0: every "
                        "fit refuses a negative value other than -1, and "
                        "-1, which requests the evidence approximation, is "
                        "no value of a box"};
  if (k == "ESIGMA")
    return OptimFitValueRule{true,
                        "ESIGMA is the energy error scale, a number > 0 "
                        "(meV/atom, or meV per sqrt(atom) under ENORM "
                        "SQRT), and every fit refuses another"};
  if (k == "WDBFILE")
    return OptimFitValueRule{true,
                        "a searched WDBFILE value is the priority of a "
                        "dataset in the fit, a number > 0 (WDBFILE 0 in the "
                        "training configuration leaves a dataset out of the "
                        "fit)"};
  if (k == "FSIGMA" || k == "SSIGMA") {
    const bool force = k == "FSIGMA";
    if (index == 1)
      return OptimFitValueRule{
          true, "the absolute part a of " + k + " is the " +
                    (force ? std::string("force") : std::string("stress")) +
                    " error scale of a structure whose " +
                    (force ? "forces are zero, a number > 0 (eV/A)"
                           : "stresses are zero, a number > 0 (GPa)") +
                    ", and every fit refuses another"};
    if (index == 2)
      return OptimFitValueRule{false, "the relative part b of " + k +
                                     " is a number >= 0, and every fit "
                                     "refuses a negative one"};
  }
  return std::nullopt;
}

std::string optim_switch_key_error(const std::string& key) {
  const std::string k = core::to_upper(key);
  if (k == "ENORM")
    return "OPTIM ENORM is not a valid search target: ENORM (ATOM or SQRT) "
           "chooses how the residual of an energy is normalised, a choice "
           "rather than a continuous parameter; the training configuration "
           "sets it.";
  if (k == "ENERGY")
    return "OPTIM ENERGY is not a valid search target: ENERGY (true or "
           "false) says whether the fit has the energies, a switch rather "
           "than a continuous parameter; the training configuration sets it.";
  return "";
}

std::string optim_retired_key_error(const std::string& key) {
  /* The keys the blueprint marks retired (BlueprintKey::isRetired: the
   * reason, the replacement and the conversion of each are its text); a
   * test holds this list to the blueprint's. */
  static const char* const retired[] = {"EWEIGHT",      "FWEIGHT",
                                        "SWEIGHT",      "WDBFILE_AUTO",
                                        "EWEIGHT_TEMP", "UNCERTAINTY",
                                        "WCOV",         "WNOISE",
                                        "ERROR",        "QUANTITY"};
  const std::string k = core::to_upper(key);
  for (const char* r : retired) {
    if (k != r) continue;
    const auto& info = core::BlueprintParser::getKeysMapRef().at(k);
    if (!info.isRetired())
      throw std::logic_error("OPTIM " + k + ": the blueprint does not mark "
                             "the key retired");
    return "OPTIM " + k + " is not a valid search target: " +
           core::retired_key_text(k, info);
  }
  return "";
}


std::string optim_integer_index_error(
    const std::string& key, const std::vector<std::string>& config,
    std::size_t pos, std::size_t index) {
  const std::string k = core::to_upper(key);
  const auto& keys = core::BlueprintParser::getKeysMapRef();
  const auto it = keys.find(k);
  if (it == keys.end() || it->second.value_type.empty() || pos >= config.size())
    return "";
  /* The key's tokens as they are match one of its formats, but not with
   * the value at pos replaced by a fraction: no format takes a fraction
   * there. */
  const std::vector<std::string>& formats = it->second.value_type;
  std::vector<std::string> fraction = config;
  fraction[pos] = "0.5";
  if (!core::BlueprintValidator::validateOnce(formats, core::join(config)) ||
      core::BlueprintValidator::validateOnce(formats, core::join(fraction)))
    return "";
  std::string takes;
  for (const std::string& f : formats)
    takes += (takes.empty() ? "" : " or ") + f;
  return "OPTIM " + k + " (" + core::to_string(index) +
         ") is not a valid search target: value " + core::to_string(index) +
         " of " + k + " is an integer (" + k + " takes " + takes +
         "), a discrete value rather than a continuous parameter; the "
         "training configuration sets it.";
}


OptimSeed prepare_optim_seed(core::Context& ctx, const std::string& command) {
  OptimSeed seed;
  seed.left_out = models::D_mJoin::expand_grids(ctx);
  /* RHOFORM (the DM_REAM density form) is mandatory and has no default:
   * read it before the OPTIM lines (non-negative AMPGRIDMB boxes) that
   * depend on it. Stored value only. */
  if (models::context_has_ream(ctx))
    seed.rho_form = models::require_rhoform(ctx, command);
  else if (models::embed_specs(ctx).size() == 1)
    seed.rho_form = models::embed_specs(ctx)[0].form;
  return seed;
}

OptimReader::OptimReader(core::Context& ctx, OptimSeed seed)
    : context(ctx), seed_(std::move(seed)) {}

OptimLine OptimReader::read(std::istringstream& iss, std::size_t nline) {
  std::string k;
  if (!(iss >> k)) {
    throw std::runtime_error("Failed to read key from input.");
  }
  /* A configuration key, and configuration keys are case-insensitive
   * (core::Context stores them upper-case): the OPTIM key is upper-cased
   * here, so optimap, the labels and every rule and lookup keyed on it
   * (below, startup_validation, the gates, the neighbour-list cutoff) see
   * one spelling whatever the case the line used. */
  k = core::to_upper(k);

  /* WEIGHTS, which each evaluation fits, and EMBF0, a choice; before
   * context.size(k) below, which would materialise the blueprint default
   * EMBF0 ZERO into the run context. */
  if (const std::string e = optim_fitted_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* Structure keys, also before context.size(k): RHOFORM has no blueprint
   * default, so size() would log an ERROR and throw a generic message. */
  if (const std::string e = optim_structure_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* Internal keys, before context.size(k), which would materialise the
   * blueprint default into the run context. */
  if (const std::string e = optim_internal_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* A retired key (the context refuses to read one) and a choice or a
   * switch of the fit, before context.size(k). */
  if (const std::string e = optim_retired_key_error(k); !e.empty())
    throw std::runtime_error(e);
  if (const std::string e = optim_switch_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* A key applied once, when the run starts and reads its data. */
  if (const std::string e = optim_startup_only_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* A legacy many-body key of a DM_Embed potential, before context.size(k),
   * which refuses an absent key without naming the component path. */
  if (const std::string e = optim_component_key_error(context, k); !e.empty())
    throw std::runtime_error(e);

  const bool named=k.find('.')!=std::string::npos;
  /* OPTIM FIXWEIGHT searches the FIXWEIGHT values, the coefficients FIXINDEX
   * names, and neither key has a default: both are in the training
   * configuration, checked before context.size(k), which would log an
   * ERROR line and refuse an absent key without saying what it is for. */
  if (k == "FIXWEIGHT") {
    const bool has_index = context.has_stored("FIXINDEX");
    const bool has_values = context.has_stored("FIXWEIGHT");
    if (!has_index && !has_values)
      throw std::runtime_error(
          "OPTIM FIXWEIGHT requires FIXINDEX and FIXWEIGHT in the training "
          "configuration (FIXINDEX names the coefficients the FIXWEIGHT "
          "values hold, and OPTIM FIXWEIGHT searches those values).");
    if (!has_values)
      throw std::runtime_error(
          "OPTIM FIXWEIGHT requires FIXWEIGHT in the training configuration "
          "(the FIXWEIGHT values hold the coefficients FIXINDEX names, and "
          "OPTIM FIXWEIGHT searches those values).");
    if (!has_index)
      throw std::runtime_error(
          "OPTIM FIXWEIGHT requires FIXINDEX in the training configuration "
          "(FIXINDEX names the coefficients the FIXWEIGHT values hold).");
  }
  /* A key of the training configuration, and one the configuration gives
   * (a line or a default), before context.size(k), which logs an ERROR line
   * and refuses either without saying what OPTIM needs. contains() inserts
   * the blueprint default, as size() does. A path of a DM_Embed parameter
   * is checked by embed_binding below. */
  if (!named) {
    const auto& keys = core::BlueprintParser::getKeysMapRef();
    if (keys.find(k) == keys.end()) {
      const std::string closest = core::BlueprintParser::findClosestKey(k, keys);
      throw std::runtime_error(
          "OPTIM " + k + ": " + k + " is not a key of the training "
          "configuration" +
          (closest.empty() ? std::string() : " (closest key: " + closest + ")"));
    }
    if (!context.contains(k))
      throw std::runtime_error(
          "OPTIM " + k + ": the training configuration has no " + k +
          " line, and OPTIM " + k + " searches the values the training "
          "configuration gives");
  }

  std::optional<models::EmbedParameterBinding> binding;
  if(named) binding=models::embed_binding(context,k);
  size_t s = named?binding->values.size():context.size(k);
  OptimLine out;
  out.key = k;
  out.named = named;
  out.line = nline;
  out.config.resize(s);   // the key's tokens are read into it below
  /* The line that selects each token of the key (0: none), across the
   * lines of the file. */
  std::vector<std::size_t>& selected = selected_[k];
  if (selected.size() < s) selected.resize(s, 0);

  /* The (indices) and the tokens after them; at least one token (AUTO, or
   * <low> <high>), and an index list of indices from 1. */
  const OptimSelector selector = read_optim_selector(iss, k);
  std::vector<size_t> user_indices = selector.indices;   // FIXWEIGHT: mapped to slots below
  for (const size_t ui : user_indices)
    if (const std::string e = optim_discrete_index_error(k, ui); !e.empty())
      throw std::runtime_error(e);

  /* Two forms after the (indices) group (read_optim_box, hpotarget_syntax.h):
   *   <low> <high> [LIN | LOG | ASINH <c>]                  explicit box
   *   AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]
   *                                                        seed-relative box
   * AUTO derives the box from the CONFIG seed of each selected index:
   *   geometric  [x0/f, x0*f] (mirrored for x0 < 0)   default for positive-only keys (RATIO 2)
   *   linear     x0 +/- max(p*|x0|, a, r*max|x0| over this OPTIM line)   otherwise (FRAC 0.5)
   *   FRAC 0 + FLOOR a  -> every index gets the same absolute half-width a (grid centres)
   * The distance cutoffs (kOptimAutoExcludedKeys: RCUT2B, RCUTMB) are
   * excluded: the neighbour list is sized from explicit high bounds. */

  /* The low-bound rule of the box (models::optim_low_bound_rule(k,
   * context)). RHOFORM NONNEG / MONOTONE (rho >= 0 by construction): the
   * AMPGRIDMB boxes are non-negative (models::optim_key_lower_bound: AUTO
   * lo = max(0, x0 - d), an explicit lo < 0 is refused), the WATOM boxes
   * positive and the SGRIDMB boxes positive under MONOTONE and, for the
   * Gaussian basis, under NONNEG. Every form: a radial bump width box
   * (SGRID2B of D2_Blip/D2_Blip5, SGRIDMB of a Blip/Blip5 density) is
   * positive. */
  const bool named_amp=named && k.size()>=11 && k.compare(k.size()-11,11,".AMPLITUDES")==0;
  const bool nonneg_amp = named ? named_amp && binding->positive :
      models::optim_key_lower_bound(k, seed_.rho_form) == models::OptimLowerBound::NonNegative;
  auto low_rule = named ? std::optional<models::OptimLowBoundRule>{} : models::optim_low_bound_rule(k, context);
  if(named && binding->positive)
    low_rule=models::OptimLowBoundRule{!named_amp,named_amp?"nonnegative amplitudes":"a positive scale or radius",k};
  /* What makes the amplitudes of a non-negative amplitude line >= 0: the
   * RHOFORM line of a DM_REAM density, or the FORM of the DENSITY block of
   * the DM_Embed component a path names (each component has its own). */
  auto form_label = [&]() -> std::string {
    if (named) {
      for (const auto& es : models::embed_specs(context))
        if (core::to_upper(es.id) == core::to_upper(binding->owner))
          return std::string("FORM ") + models::rhoform_name(es.form) +
                 " of the DENSITY of DESCRIPTOR " + es.id;
    } else if (seed_.rho_form) {
      return std::string("RHOFORM ") + models::rhoform_name(*seed_.rho_form);
    }
    throw std::logic_error("OPTIM " + k + ": a non-negative amplitude line "
                           "without a density form");
  };

  /* The box as the line writes it: its syntax, which needs no training
   * configuration (the AUTO options, the tag, the explicit bounds). */
  const OptimBoxSyntax box = read_optim_box(selector.tokens, k);
  const bool auto_box = box.auto_box;
  const double low = box.low, high = box.high;
  const std::string& low_tok = box.low_tok;   // the explicit bounds as written
  const std::string& high_tok = box.high_tok;
  const std::string& tag_c_tok = box.c_tok;   // the c of ASINH <c>, as written
  struct AutoRule { bool geometric = false; double factor = 0.0; double floor = 0.0; double group_floor = 0.0; } rule;
  rule.geometric = box.geometric;
  rule.factor = box.factor;
  rule.floor = box.floor;
  rule.group_floor = box.group_floor;
  const std::string& factor_tok = box.factor_tok;   // the RATIO or FRAC value as written
  const std::string& group_floor_tok = box.group_floor_tok;
  const bool floor_set = box.floor_set;   // FLOOR or GROUP_FLOOR given
  const bool mode_set = box.mode_set;     // RATIO or FRAC given
  out.shape = box.shape;
  out.c = box.c;
  out.tagged = box.tagged;
  out.auto_box = auto_box;
  if (auto_box) {
    if (std::find(kOptimAutoExcludedKeys.begin(), kOptimAutoExcludedKeys.end(),
                  k) != kOptimAutoExcludedKeys.end())
      throw std::runtime_error("OPTIM " + k + ": a cutoff key takes explicit "
                               "bounds, OPTIM " + k + " (indices) <low> <high>, "
                               "and no AUTO box: the neighbour list is sized "
                               "from the explicit high bound");
    if (!mode_set) {
      rule.geometric = named ? binding->positive && !named_amp : optim_key_positive_only(k);
      rule.factor    = rule.geometric ? 2.0 : 0.5;
    }
    /* FLOOR and GROUP_FLOOR bound the half-width of the linear rule; the
     * geometric box [x0/f, x0*f] has none, so they would be ignored. A
     * value the fit takes only > 0 (ESIGMA, FSIGMA (1), SSIGMA (1),
     * WDBFILE: optim_fit_value_rule) has a geometric box by default, the
     * relative part FSIGMA (2) or SSIGMA (2) a linear one: the default is
     * that of each index of the line. */
    std::string geometric_default;  // the key (and index) of such a box
    if (mode_set) {
      if (rule.geometric) geometric_default = k;
    } else if (rule.geometric) {
      geometric_default = k;
    } else if (!named) {
      for (const std::size_t ui : user_indices) {
        const auto v = optim_fit_value_rule(k, ui);
        if (!v || !v->strict) continue;
        geometric_default =
            (k == "FSIGMA" || k == "SSIGMA") ? k + " (" + core::to_string(ui) + ")"
                                             : k;
        break;
      }
    }
    if (!geometric_default.empty() && floor_set)
      throw std::runtime_error(
          "OPTIM " + geometric_default + ": FLOOR and GROUP_FLOOR set the smallest "
          "half-width of an AUTO FRAC box; " +
          (mode_set ? "AUTO RATIO " + factor_tok + " makes the geometric "
                      "box [seed/" + factor_tok + ", seed*" + factor_tok +
                      "], which they do not change, so they are given with "
                      "FRAC <p> only"
                    : "the AUTO box of " + geometric_default + " is geometric by default "
                      "(RATIO 2), which they do not change, so they are "
                      "given with FRAC <p> only"));
    /* Non-negative amplitudes: NNLS fits put amplitudes at exactly 0, so a
     * linear rule without a floor gets the refit writer's group floor
     * (1e-3 max|x0| of the line) and the box [0, d] for a zero seed. */
    if (nonneg_amp && !rule.geometric && !floor_set) rule.group_floor = 1e-3;
  }

  /* The indices as the line wrote them (for FIXWEIGHT, model weights). */
  const std::vector<size_t> typed_indices = user_indices;
  if (k == "FIXWEIGHT") {
    /* OPTIM FIXWEIGHT indices name coefficients (those FIXINDEX lists),
     * not positions in the FIXWEIGHT value list. Map each user index to
     * its FIXWEIGHT slot through FIXINDEX; declaration order defines the
     * pairing. Back-compatible with `tadah refit` targets, which list
     * FIXINDEX 1-K so the coefficient index == slot. (The training
     * configuration has FIXINDEX and FIXWEIGHT: checked above.) */
    /* core::parse_fixindex, as the trainers read FIXINDEX; the weight
     * count is not known here (startup validation checks the bounds). */
    std::vector<std::string> fi_str(context.size("FIXINDEX"));
    context.get("FIXINDEX", fi_str);
    const std::vector<size_t> fixidx =
        core::parse_fixindex(context, std::nullopt);
    for (auto& ui : user_indices) {
      auto it = std::find(fixidx.begin(), fixidx.end(), ui);
      if (it == fixidx.end()) {
        throw std::runtime_error(
          "OPTIM FIXWEIGHT (" + core::to_string(ui) + "): coefficient " +
          core::to_string(ui) + " is not in FIXINDEX (" + core::join(fi_str) +
          "); the indices of OPTIM FIXWEIGHT name coefficients, those "
          "FIXINDEX lists, not positions among the FIXWEIGHT values");
      }
      ui = static_cast<size_t>(it - fixidx.begin()) + 1;  // 1-based slot
    }
  }

  if(named) {
    out.config.clear();
    for(double v:binding->values) out.config.push_back(core::format_roundtrip(v));
  } else context.get<std::vector<std::string>>(k, out.config);

  /* LAMBDA -1 requests the evidence approximation, which finds the penalty
   * in each fit: a searched LAMBDA would replace it, and the starting
   * point -1 is no value of a box. */
  if (k == "LAMBDA" && !out.config.empty() &&
      core::is_number(out.config[0]) &&
      core::to_double(out.config[0]) == -1.0)
    throw std::runtime_error(
        "OPTIM LAMBDA: the training configuration has LAMBDA -1, which "
        "requests the evidence approximation: each fit then finds the "
        "penalty itself, and OPTIM LAMBDA would replace it with the values "
        "of the box. LAMBDA >= 0 in the training configuration is the "
        "starting point of a search of LAMBDA; without OPTIM LAMBDA the "
        "evidence approximation finds it in every evaluation.");

  /* The checks a box [lo_b, hi_b] of the index @p ui (a FIXWEIGHT slot)
   * meets below, past its AUTO derivation: what the fit takes of the value
   * (optim_fit_value_rule), the rank threshold of OALGO and the low-bound
   * rule of the key. A hint of a message names a box only when the line's
   * own checks take it. */
  /* The low-bound rule of the index @p ui: on a D2_mJoin line, that of the
   * descriptor whose entry it is (a D2_BP's width has none). */
  auto entry_low_rule = [&](std::size_t ui) {
    if (!named && low_rule)
      return models::optim_low_bound_rule(k, context, ui - 1);
    return low_rule;
  };
  auto box_passes = [&](double lo_b, double hi_b, std::size_t ui) {
    const std::optional<OptimFitValueRule> vr =
        named ? std::nullopt : optim_fit_value_rule(k, ui);
    if (vr && vr->strict && !(lo_b > 0.0)) return false;
    if (vr && !vr->strict && lo_b < 0.0) return false;
    if (!named && k == "OALGO" && ui == 2 && !(lo_b > 0.0 && hi_b < 1.0))
      return false;
    if (const auto r = entry_low_rule(ui); r && !r->admits(lo_b))
      return false;
    return true;
  };
  /* An AUTO FRAC box of the value: clamped at 0 for a non-negative
   * amplitude and for a value the fit takes >= 0 other than LAMBDA. */
  auto frac_clamped = [&](const std::optional<OptimFitValueRule>& vr) {
    return nonneg_amp || (vr && !vr->strict && k != "LAMBDA");
  };
  /* Whether an AUTO FRAC box with a FLOOR above 0 around the seed x0 of
   * the index passes the line's checks. */
  auto floor_box_passes = [&](double x0, std::size_t ui,
                              const std::optional<OptimFitValueRule>& vr) {
    const double a = 0.5;  // a FLOOR above 0
    double lo_b = x0 - a;
    if (frac_clamped(vr) && lo_b < 0.0) lo_b = 0.0;
    return box_passes(lo_b, x0 + a, ui);
  };
  /* The explicit bounds that give the value of index @p ui (written
   * @p typed) a box the line's checks take: from 0 when a box from 0
   * passes, above 0 otherwise. */
  auto explicit_hint = [&](std::size_t ui, std::size_t typed) {
    const std::string line = "OPTIM " + k + " (" + core::to_string(typed) + ")";
    if (box_passes(0.0, 0.5, ui))
      return "; the explicit bounds `" + line + " 0 <high>` give it a box";
    return "; the explicit bounds `" + line + " <low> <high>` with <low> > 0 "
           "give it a box";
  };

  /* An explicit box tagged LOG reaches 0 or below, or one that ASINH <c>
   * maps to an empty or infinite range: refused at the line (SearchSpace
   * keeps both checks as its own invariants). */
  if (!auto_box) {
    if (box.shape == OptimShape::Log && !(low > 0.0)) {
      bool every_index_takes_it = true;
      for (const size_t ui : user_indices)
        every_index_takes_it = every_index_takes_it && box_passes(low, high, ui);
      throw std::runtime_error(
          "OPTIM " + k + ": the line is tagged LOG, and <low> " + low_tok +
          " is not > 0, and a box tagged LOG lies above 0" +
          (every_index_takes_it
               ? std::string("; a line without a tag, or tagged LIN or "
                             "ASINH <c>, takes this box")
               : std::string()));
    }
    if (box.shape == OptimShape::Asinh) {
      const double tlo = std::asinh(low / box.c), thi = std::asinh(high / box.c);
      if (!value_is_finite(tlo) || !value_is_finite(thi) ||
          !(thi > tlo))
        throw std::runtime_error(
            "OPTIM " + k + ": under ASINH " + tag_c_tok + " the box [" +
            low_tok + ", " + high_tok + "] maps to [" +
            core::format_roundtrip(tlo) + ", " + core::format_roundtrip(thi) +
            "], which has no finite width; a c of the size of the values of "
            "the box gives one");
    }
  }

  /* AUTO: the group floor is r * max|x0| over the indices of THIS line. */
  double group_max = 0.0;
  if (auto_box) {
    for (const auto& ui : user_indices) {
      const size_t i = optim_value_position(out.config, ui);
      if (i < s && core::is_number(out.config[i]))
        group_max = std::max(group_max, std::fabs(core::to_double(out.config[i])));
    }
  }
  /* A geometric box cannot reach 0: non-negative amplitudes take RATIO only
   * when every seed of the line is > 0. */
  if (auto_box && nonneg_amp && rule.geometric) {
    for (std::size_t j = 0; j < user_indices.size(); ++j) {
      const size_t i = optim_value_position(out.config, user_indices[j]);
      if (i >= s || !core::is_number(out.config[i])) continue;  // reported below
      if (core::to_double(out.config[i]) > 0.0) continue;
      throw std::runtime_error(
          "OPTIM " + k + " (" + core::to_string(typed_indices[j]) +
          "): AUTO RATIO under " + form_label() + " needs every seed on the "
          "line > 0 (got " + out.config[i] + "); FRAC, the default, gives a "
          "box clamped at 0");
    }
  }

  /* The numeric values of the key, those its indices count. */
  std::size_t numeric_values = 0;
  for (const auto& t : out.config)
    if (core::is_number(t)) ++numeric_values;

  for (std::size_t j = 0; j < user_indices.size(); ++j) {
    const size_t ui = user_indices[j];
    {
      const size_t i = optim_value_position(out.config, ui);
      if (const std::string e =
              optim_integer_index_error(k, out.config, i, typed_indices[j]);
          !e.empty())
        throw std::runtime_error(e);
    }
    /* A refusal of this entry says why; the catch below names the entry
     * once, "OPTIM <KEY> (<i>): <why>". */
    try {
      size_t i = optim_value_position(out.config, ui);
      if (i>=s) {
        /* Values of a potential its descriptors do not read were left out
         * (D_mJoin::expand_grids): the indices address the values kept. */
        std::string why;
        bool left_out = false;
        for (const auto& lo : seed_.left_out) {
          if (lo.key != k) continue;
          left_out = true;
          std::vector<std::string> out, rest = lo.kept;
          for (const auto& v : lo.given) {
            const auto it = std::find(rest.begin(), rest.end(), v);
            if (it != rest.end()) rest.erase(it);
            else out.push_back(v);
          }
          why = " (of the potential's " + k + " " + core::join(lo.given) +
                ", the descriptors do not read " + core::join(out) +
                (out.size() == 1 ? ", which was" : ", which were") +
                " left out, and OPTIM indices address the values kept, " +
                core::join(lo.kept);
          if (ui >= 1 && ui <= lo.given.size()) {
            const std::string& v = lo.given[ui - 1];
            const auto it = std::find(lo.kept.begin(), lo.kept.end(), v);
            if (it != lo.kept.end())
              why += "; the potential's value at index " + core::to_string(ui) +
                     ", " + v + ", is index " +
                     core::to_string(static_cast<std::size_t>(
                         it - lo.kept.begin()) + 1);
          }
          why += ")";
        }
        /* FSIGMA and SSIGMA of one value: the relative part b is index 2,
         * which the configuration holds once it gives both values. */
        if (why.empty() && (k == "FSIGMA" || k == "SSIGMA") && ui == 2 &&
            s == 1)
          why = " (its absolute part a; its relative part b, index 2, is "
                "searched when the training configuration gives both, " + k +
                " <a> <b>)";
        const std::string has =
            numeric_values == 1
                ? "1 value"
                : core::to_string(numeric_values) + " values";
        const std::string range =
            numeric_values == 1
                ? "the index of OPTIM " + k + " is 1"
                : "the indices of OPTIM " + k + " run from 1 to " +
                      core::to_string(numeric_values);
        /* A FIXWEIGHT slot is an entry of FIXINDEX, which
         * core::parse_fixindex pairs with the FIXWEIGHT values one to
         * one. */
        if (k == "FIXWEIGHT")
          throw std::logic_error(
              "OPTIM FIXWEIGHT: coefficient " +
              core::to_string(typed_indices[j]) + " is entry " +
              core::to_string(ui) + " of FIXINDEX, past the FIXWEIGHT "
              "values");
        if (numeric_values == 0)
          throw std::runtime_error(
              k + " has no numeric value in the training configuration, so "
              "no index of OPTIM " + k + " selects one");
        throw std::runtime_error(
            k + " has " + has +
            (left_out ? " that the descriptors read"
                      : " in the training configuration") +
            ", so " + range + why);
      }
      /* One OPTIM line per index: a second box for an index would replace
       * the first without a word. Lines of one key that select other
       * indices add to it (in any case, the key being upper-cased). */
      if (selected[i] != 0)
        throw std::runtime_error(
            "the OPTIM " + k + " line on line " + core::to_string(selected[i]) +
            " selects this index as well; one OPTIM line selects each index "
            "(OPTIM keys match in any case)");
      double lo_i = low, hi_i = high;
      OptimBoxOrigin origin = OptimBoxOrigin::Explicit;
      /* What the fit takes of this value (> 0, >= 0), whose box keeps to
       * it: the AUTO box of a value > 0 is geometric unless RATIO or FRAC
       * says otherwise, that of a value >= 0 linear and clamped at 0. */
      const std::optional<OptimFitValueRule> value_rule =
          named ? std::nullopt : optim_fit_value_rule(k, ui);
      const bool geometric =
          mode_set ? rule.geometric
                   : rule.geometric || (value_rule && value_rule->strict);
      const double factor =
          mode_set ? rule.factor : (geometric ? 2.0 : 0.5);
      const std::string factor_text =
          mode_set ? factor_tok : core::format_roundtrip(factor);
      const std::size_t typed = typed_indices[j];
      if (auto_box) {
        /* A number: ui_offset gives the position of the ui-th number of
         * the key (ui >= 1, index 0 refused by the selector). */
        const double x0 = core::to_double(out.config[i]);
        const std::string& seed = out.config[i];
        if (nonneg_amp && x0 < 0.0)
          throw std::runtime_error("the seed " + seed + " is negative, but " +
                                   form_label() + " requires every AMPGRIDMB "
                                   "amplitude >= 0");
        /* A value every fit takes >= 0 (optim_fit_value_rule) is the seed
         * of its box: a negative one is refused as such, before a FRAC box
         * clamped at 0 would be reversed ([0, x0 + d] with x0 + d < 0) or
         * leave the seed out ([0, x0 + d] with x0 + d > 0). */
        if (value_rule && !value_rule->strict && x0 < 0.0)
          throw std::runtime_error("the seed " + seed + " is negative, but " +
                                   value_rule->why);
        if (geometric) {
          if (!(x0 != 0.0)) {
            /* The key, with the index for FSIGMA and SSIGMA, whose two
             * values have different defaults. */
            const std::string subject =
                (k == "FSIGMA" || k == "SSIGMA")
                    ? k + " (" + core::to_string(typed) + ")"
                    : k;
            const std::string head =
                mode_set
                    ? "AUTO RATIO " + factor_text + " gives the box [seed/" +
                          factor_text + ", seed*" + factor_text +
                          "], which needs a seed other than 0, and the seed "
                          "is 0"
                    : "AUTO without RATIO or FRAC gives " + subject +
                          " the box [seed/2, seed*2] (RATIO 2), which needs a "
                          "seed other than 0, and the seed is 0";
            throw std::runtime_error(
                head + (floor_box_passes(x0, ui, value_rule)
                            ? std::string("; AUTO FRAC <p> FLOOR <a> gives a "
                                          "box of half-width a or more "
                                          "around it")
                            : explicit_hint(ui, typed)));
          }
          const double a = x0 / factor, b = x0 * factor;
          lo_i = std::min(a, b); hi_i = std::max(a, b);
          origin = OptimBoxOrigin::AutoGeometric;
          if (!(hi_i > lo_i))
            throw std::runtime_error(
                "the AUTO box [" + core::format_roundtrip(lo_i) + ", " +
                core::format_roundtrip(hi_i) + "] around the seed " + seed +
                " has zero width in double precision; a larger RATIO gives it "
                "a width");
        } else {
          const double d = std::max({factor * std::fabs(x0), rule.floor,
                                     rule.group_floor * group_max});
          /* d is 0 only when every seed of the line is 0: the group floor
           * 1e-3 is relative to the largest of them. */
          if (!(d > 0.0) && nonneg_amp && !floor_set)
            throw std::runtime_error(
                "every seed on this line is 0, so the AUTO box has zero "
                "width (FRAC and the GROUP_FLOOR 1e-3 that " + form_label() +
                " implies for " + k + " are relative to the seeds); AUTO "
                "FLOOR <a> with a > 0 gives the box [0, a], and the explicit "
                "bounds `OPTIM " + k + " (" + core::to_string(typed) +
                ") 0 <high>` the box [0, <high>]");
          if (!(d > 0.0)) {
            std::string head;
            std::string hint;
            if (group_max > 0.0) {
              head = "AUTO FRAC gives a zero-width box (the seed " + seed +
                     " times FRAC " + factor_text + " is 0)";
              hint = "; FLOOR <a> or GROUP_FLOOR <r> above 0 gives it a "
                     "smallest half-width";
            } else {
              head = "every seed on this line is 0, so AUTO FRAC gives a "
                     "zero-width box";
              if (rule.group_floor > 0.0)
                head += ", and GROUP_FLOOR " + group_floor_tok + ", a "
                        "fraction of the largest seed on the line, adds "
                        "nothing";
              hint = "; FLOOR <a> above 0 gives it the half-width a";
            }
            if (!floor_box_passes(x0, ui, value_rule))
              hint = explicit_hint(ui, typed);
            throw std::runtime_error(head + hint);
          }
          lo_i = x0 - d; hi_i = x0 + d;
          origin = OptimBoxOrigin::AutoLinear;
          /* non-negative amplitudes: lo = max(0, x0 - d), before the LOG
           * check below (a LOG box cannot start at 0) */
          if (frac_clamped(value_rule)) {
            if (lo_i < 0.0) lo_i = 0.0;
            origin = OptimBoxOrigin::AutoLinearNonneg;
          }
          /* A half-width below the spacing of the doubles at the seed
           * rounds the box to the seed (the seed of a clamped box is >= 0:
           * a negative one is refused above). */
          if (!(hi_i > lo_i))
            throw std::runtime_error(
                "the AUTO box around the seed " + seed + " has zero width in "
                "double precision (half-width " + core::format_roundtrip(d) +
                "); a larger FRAC or FLOOR gives it a width");
        }
        /* A box past the largest double (x0*f, x0 + d). */
        if (!value_is_finite(lo_i) || !value_is_finite(hi_i))
          throw std::runtime_error(
              "the AUTO box [" + core::format_roundtrip(lo_i) + ", " +
              core::format_roundtrip(hi_i) + "] around the seed " + seed +
              " is not finite; a smaller RATIO or FRAC gives a finite box");
        if (box.shape == OptimShape::Log && lo_i <= 0.0)
          throw std::runtime_error(
              "the line is tagged LOG, and the AUTO box [" +
              core::format_roundtrip(lo_i) + ", " +
              core::format_roundtrip(hi_i) + "] reaches 0 or below, and a box "
              "tagged LOG lies above 0" +
              (x0 > 0.0 ? std::string("; AUTO RATIO <f> gives a box above 0 "
                                      "around this seed")
                        : std::string()) +
              (box_passes(lo_i, hi_i, ui)
                   ? std::string("; a line without a tag, or tagged LIN or "
                                 "ASINH <c>, takes this box")
                   : std::string()));
      }
      /* A value of the fit a fit refuses at or below 0, or below 0
       * (optim_fit_value_rule): every such candidate would be refused by
       * its evaluation. */
      if (value_rule && value_rule->strict && !(lo_i > 0.0))
        throw std::runtime_error("the low bound " +
                                 core::format_roundtrip(lo_i) +
                                 " is not > 0, but " + value_rule->why +
                                 ", so the low bound of its box is > 0");
      if (value_rule && !value_rule->strict && lo_i < 0.0)
        throw std::runtime_error("the low bound " +
                                 core::format_roundtrip(lo_i) +
                                 " is negative, but " + value_rule->why +
                                 ", so the low bound of its box is >= 0");
      /* OALGO's second value, the rank threshold M of OALGO 1 and 3: only
       * 0 < M < 1 sets a threshold (below 0 OALGO 1 takes the machine
       * precision, at 0 or below OALGO 3 leaves out only a singular value
       * of 0, whatever M), and a fit refuses 1 or more. */
      if (!named && k == "OALGO" && ui == 2 && !(lo_i > 0.0 && hi_i < 1.0))
        throw std::runtime_error(
            "the box [" + core::format_roundtrip(lo_i) + ", " +
            core::format_roundtrip(hi_i) +
            "] is not inside (0, 1), but the second value of OALGO is the "
            "rank threshold M of OALGO 1 and 3, which only a value 0 < M < "
            "1 sets: every M below 0 is the machine precision under OALGO "
            "1, every M of 0 or below leaves out only a singular value of 0 "
            "under OALGO 3, and a fit refuses an M of 1 or more, so the box "
            "of a search is inside (0, 1)");
      /* The low-bound rule of this entry (entry_low_rule: on a D2_mJoin
       * line, that of the descriptor whose entry it is): an explicit low
       * bound as written, an AUTO box with both ends (the line wrote
       * neither). */
      if (const auto entry_rule = entry_low_rule(ui);
          entry_rule && !entry_rule->admits(lo_i)) {
        const std::string requires_ =
            entry_rule->by + " requires " + entry_rule->requirement;
        if (auto_box)
          throw std::runtime_error(
              "the AUTO box [" + core::format_roundtrip(lo_i) + ", " +
              core::format_roundtrip(hi_i) + "] has the low bound " +
              core::format_roundtrip(lo_i) + ", which " +
              (entry_rule->strict ? "is not > 0" : "is negative") + ", but " +
              requires_ + ", so the low bound of its box is " +
              (entry_rule->strict ? "> 0" : ">= 0"));
        throw std::runtime_error(
            "the low bound " + low_tok +
            (entry_rule->strict ? " is not > 0" : " is negative") + ", but " +
            requires_ + ", so the low bound of its box is " +
            (entry_rule->strict ? std::string("> 0")
                                : std::string(">= 0 (its AUTO FRAC box, the "
                                              "default, is clamped at 0)")));
      }
      selected[i] = nline;
      out.boxes.push_back({typed, i, lo_i, hi_i, origin});
    } catch (const std::exception& e) {
      throw std::runtime_error("OPTIM " + k + " (" +
                               core::to_string(typed_indices[j]) + "): " +
                               e.what());
    }
  }
  /* Every box has high > low: an explicit box is checked at its bounds, an
   * AUTO box where it is derived. */
  for (const OptimBox& b : out.boxes)
    if (!(b.hi > b.lo))
      throw std::logic_error("OPTIM " + k + ": a box with high <= low passed "
                             "the checks of its line");
  return out;
}

std::vector<OptimTargetLine> read_optim_lines(std::istream& in,
                                              core::Context& ctx,
                                              const OptimSeed& seed) {
  OptimReader reader(ctx, seed);
  std::vector<OptimTargetLine> out;
  for (const TargetLine& l : target_logical_lines(in)) {
    if (l.unfinished)
      throw TargetLineError(l.last, l.text, kTargetUnfinishedLineMsg);
    if (l.key != "OPTIM") continue;
    std::istringstream rest(l.text);
    std::string word;
    rest >> word;
    try {
      out.push_back({l, reader.read(rest, l.last)});
    } catch (const std::exception& e) {
      throw TargetLineError(l.last, l.text, e.what());
    }
  }
  return out;
}

}  // namespace mlip
}  // namespace tadah
