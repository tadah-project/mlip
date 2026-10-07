#include <tadah/mlip/weight_provisioning.h>

#include <tadah/mlip/models/basis.h>

#include <tadah/core/core_types.h>
#include <tadah/core/number_format.h>
#include <tadah/core/utils.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

constexpr const char *kLoneKeyMsg =
    "FIXINDEX requires FIXWEIGHT (and vice versa). Provide both keys with "
    "matching lengths; the i-th FIXWEIGHT value pins the weight named by "
    "the i-th FIXINDEX entry. There is no auto-seeding from WEIGHTS.";

struct FixedSpec {
  std::vector<size_t> indices;  // 1-based, declaration order, deduplicated
  std::vector<double> values;   // values[k] pins weight indices[k]
};

/* Parse and validate FIXINDEX/FIXWEIGHT. Both keys must be present. */
FixedSpec parse_fixed_spec(const tadah::core::Context &ctx,
                           std::size_t n_cols) {
  FixedSpec spec;
  spec.indices = tadah::core::parse_indices(ctx("FIXINDEX"));

  const std::vector<std::string> &fw = ctx("FIXWEIGHT");
  if (fw.size() != spec.indices.size()) {
    throw std::runtime_error(
        "FIXINDEX names " + std::to_string(spec.indices.size()) +
        " weights but FIXWEIGHT provides " + std::to_string(fw.size()) +
        " values; they must pair 1:1 (the i-th value pins the i-th index).");
  }

  for (size_t idx : spec.indices) {
    if (idx < 1 || idx > n_cols) {
      throw std::runtime_error(
          "FIXINDEX entry " + std::to_string(idx) +
          " is out of range: this model has " + std::to_string(n_cols) +
          " weights (valid range 1.." + std::to_string(n_cols) + ").");
    }
  }

  spec.values.reserve(fw.size());
  for (const std::string &v : fw) {
    try {
      // to_double: std::stod that also accepts a subnormal value.
      spec.values.push_back(tadah::core::to_double(v));
    } catch (const std::exception &) {
      throw std::runtime_error("FIXWEIGHT entry is not a number: '" + v + "'");
    }
  }
  return spec;
}

/* Compress the weight indices in 1..n_cols NOT covered by `indices` into a
 * compact "1,3-5,9" style list for error messages. */
std::string format_uncovered(const std::vector<size_t> &indices,
                             std::size_t n_cols) {
  std::vector<bool> covered(n_cols + 1, false);
  for (size_t idx : indices) covered[idx] = true;

  std::ostringstream oss;
  bool first = true;
  size_t i = 1;
  while (i <= n_cols) {
    if (covered[i]) { ++i; continue; }
    size_t j = i;
    while (j + 1 <= n_cols && !covered[j + 1]) ++j;
    if (!first) oss << ",";
    if (j > i) oss << i << "-" << j;
    else oss << i;
    first = false;
    i = j + 1;
  }
  return oss.str();
}

}  // namespace

bool basis_drawn_from_training_data(const tadah::core::Context &ctx) {
  if (!ctx.contains("MODEL") || ctx.size("MODEL") < 2) return false;
  return ctx.get<std::string>("MODEL", 0) == "M_KRR" &&
         ctx.get<std::string>("MODEL", 1) != "Kern_Linear";
}

int krr_implementation(const tadah::core::Context &ctx) {
  return ctx.contains("MODEL") && ctx.size("MODEL") >= 3
             ? ctx.get<int>("MODEL", 2)
             : 1;
}

std::size_t model_weight_count(DM_Function_Base &fb,
                               const tadah::core::Context &ctx) {
  if (!basis_drawn_from_training_data(ctx)) return fb.get_phi_cols(ctx);
  const std::string model = "M_KRR " + ctx.get<std::string>("MODEL", 1);
  // The bound of the basis draw (Basis::check_sbasis): the intercept
  // vectors e_0 .. e_{n0-1}, then at least one vector drawn from the
  // training AEDs (implementation 2: the structures' mean AEDs).
  const std::size_t n0 = n_intercept_basis_vectors(ctx);
  const std::string what = krr_implementation(ctx) == 2
                               ? "training structures"
                               : "training AEDs";
  if (!ctx.contains("SBASIS"))
    throw std::runtime_error(
        model + " needs SBASIS, the number of basis vectors training draws "
        "(" + intercept_basis_vectors_text(n0) + ", the others are drawn "
        "from the " + what + ").");
  const std::size_t s = ctx.get<std::size_t>("SBASIS");
  const std::string too_few = sbasis_lower_bound_problem(s, n0, what);
  if (!too_few.empty()) throw std::runtime_error(model + ": " + too_few);
  return s;
}

WeightSource classify_weight_source(const tadah::core::Context &ctx,
                                    std::size_t n_cols) {
  const bool has_db = ctx.contains("DBFILE") && ctx.size("DBFILE") > 0;
  const bool has_fw = ctx.contains("FIXWEIGHT");
  const bool has_fi = ctx.contains("FIXINDEX");

  if (has_fw != has_fi) {
    throw std::runtime_error(kLoneKeyMsg);
  }

  if (!has_db && basis_drawn_from_training_data(ctx)) {
    throw std::runtime_error(
        "No training DBFILE: M_KRR " + ctx.get<std::string>("MODEL", 1) +
        " draws its SBASIS basis vectors from the training data, and its "
        "weights belong to them. Provide a training DBFILE.");
  }

  if (!has_fw) {
    if (!has_db) {
      throw std::runtime_error(
          "No training DBFILE and no FIXINDEX/FIXWEIGHT: the model weights "
          "cannot be determined. Provide a DBFILE for least-squares "
          "training, or pin every weight with FIXINDEX/FIXWEIGHT.");
    }
    return WeightSource::FullLS;
  }

  const FixedSpec spec = parse_fixed_spec(ctx, n_cols);
  const bool full_coverage = spec.indices.size() == n_cols;

  if (!has_db) {
    if (!full_coverage) {
      throw std::runtime_error(
          "No training DBFILE: every model weight must be pinned. Weights " +
          format_uncovered(spec.indices, n_cols) +
          " have no training data to be fitted — add them to "
          "FIXINDEX/FIXWEIGHT (optionally with OPTIM FIXWEIGHT lines to "
          "search them).");
    }
    return WeightSource::FixedOnlyNoData;
  }

  return full_coverage ? WeightSource::AllFixedData : WeightSource::PartialLS;
}

void provision_fixed_weights(M_Tadah_Base &model,
                             const tadah::core::Context &ctx,
                             std::size_t n_cols) {
  if (basis_drawn_from_training_data(ctx)) {
    throw std::runtime_error(
        "provision_fixed_weights: M_KRR " + ctx.get<std::string>("MODEL", 1) +
        " has no basis vectors before training (they are drawn from the "
        "training data), so its weights cannot be set from FIXWEIGHT alone; "
        "train it (FIXINDEX/FIXWEIGHT then pin weights of the drawn basis).");
  }
  const bool has_fw = ctx.contains("FIXWEIGHT");
  const bool has_fi = ctx.contains("FIXINDEX");
  if (!has_fw || !has_fi) {
    throw std::runtime_error(kLoneKeyMsg);
  }

  const FixedSpec spec = parse_fixed_spec(ctx, n_cols);
  if (spec.indices.size() != n_cols) {
    throw std::runtime_error(
        "provision_fixed_weights: every model weight must be pinned. "
        "Weights " + format_uncovered(spec.indices, n_cols) +
        " are not covered by FIXINDEX.");
  }

  tadah::core::t_type w(n_cols);
  for (size_t k = 0; k < spec.indices.size(); ++k) {
    w[spec.indices[k] - 1] = spec.values[k];
  }
  model.set_weights(w);
  model.trained = true;
}

}  // namespace mlip
}  // namespace tadah
