#include <tadah/mlip/intercepts.h>

#include <tadah/core/intercept_layout.h>
#include <tadah/core/utils.h>

#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

std::string atom_list(const tadah::core::Context &ctx) {
  std::string s;
  if (!ctx.has_stored("ATOM")) return s;
  for (std::size_t i = 0; i < ctx.size("ATOM"); ++i)
    s += (i ? " " : "") + ctx.get<std::string>("ATOM", i);
  return s;
}

std::string plural(std::size_t n, const std::string &word) {
  return std::to_string(n) + " " + word + (n == 1 ? "" : "s");
}

}  // namespace

void check_bias_constraints(const tadah::core::Context &ctx) {
  const bool norm = ctx.contains("NORM") && ctx.get<bool>("NORM");
  const bool bias = ctx.contains("BIAS") && ctx.get<bool>("BIAS");
  const std::string model_bf =
      ctx.contains("MODEL") && ctx.size("MODEL") >= 2
          ? ctx.get<std::string>("MODEL", 1)
          : std::string{};

  if (norm && !bias && (model_bf == "BF_Linear" || model_bf == "Kern_Linear")) {
    throw std::runtime_error(
        "Configuration: NORM=true with MODEL=" + model_bf +
        " requires BIAS=true. The weights fitted to the normalised "
        "descriptors are converted to weights of the raw descriptors, and "
        "the constant this moves (the descriptor means times the weights) "
        "goes into the intercept columns; without them the potential would "
        "predict wrong energies (tadah predict and LAMMPS). Set BIAS=true or "
        "NORM=false.");
  }
  if (bias && (!ctx.contains("ATOM") || ctx.size("ATOM") == 0)) {
    throw std::runtime_error(
        "Configuration: BIAS=true requires a non-empty ATOM list "
        "(BIAS adds one intercept column per species).");
  }
}

std::string fixindex_intercept_warning(const tadah::core::Context &ctx) {
  if (!ctx.has_stored("FIXINDEX")) return "";
  const std::size_t n = tadah::core::n_intercept_columns(ctx);
  if (n < 2) return "";
  const std::string N = std::to_string(n);
  return "FIXINDEX with BIAS true and " + N + " ATOM species (" +
         atom_list(ctx) + "): weights 1 to " + N +
         " are the per-species intercepts, in ATOM order, and the descriptor "
         "weights start at index " + std::to_string(n + 1) +
         ". Before 1.6.0 BIAS true had one intercept shared by all species "
         "and the descriptor weights started at index 2: an index written "
         "for that layout now pins another weight (descriptor weight k moved "
         "from index k + 1 to k + " + N + ").";
}

void throw_element_not_in_atom(const tadah::core::Element &e,
                               const tadah::core::Context &ctx,
                               const std::string &where) {
  throw std::runtime_error(
      "BIAS true: " + where + " has an atom of " + e.symbol +
      ", which has no intercept column: ATOM " + atom_list(ctx) +
      " does not list " + e.symbol +
      " (BIAS gives one intercept column per ATOM species). Use a "
      "potential trained with " + e.symbol + ", or leave such structures "
      "out.");
}

void require_intercepts_determined(const StructureDB &stdb,
                                   const tadah::core::Context &ctx) {
  const std::size_t n_int = tadah::core::n_intercept_columns(ctx);
  if (n_int == 0) return;
  const auto col = tadah::core::intercept_columns_by_Z(ctx);

  // Intercepts that FIXINDEX/FIXWEIGHT pin are not fitted.
  std::vector<bool> pinned(n_int, false);
  if (ctx.has_stored("FIXINDEX") && ctx.has_stored("FIXWEIGHT")) {
    std::vector<std::string> fi(ctx.size("FIXINDEX"));
    ctx.get("FIXINDEX", fi);
    for (std::size_t i : tadah::core::parse_indices(fi))
      if (i >= 1 && i <= n_int) pinned[i - 1] = true;
  }
  std::vector<std::size_t> fitted;
  for (std::size_t k = 0; k < n_int; ++k)
    if (!pinned[k]) fitted.push_back(k);
  if (fitted.empty()) return;

  // The distinct species-count vectors of the energy-weighted structures.
  const double eweightglob = ctx.get<double>("EWEIGHT");
  std::map<std::vector<std::size_t>, std::size_t> compositions;
  std::size_t n_weighted = 0;
  for (std::size_t s = 0; s < stdb.size(); ++s) {
    const Structure &st = stdb(s);
    std::vector<std::size_t> counts(n_int, 0);
    for (const Atom &a : st.atoms) {
      const int c = a.Z >= 0 && a.Z < static_cast<int>(col.size()) ? col[a.Z] : -1;
      if (c < 0)
        throw_element_not_in_atom(a, ctx,
                                  "training structure " +
                                      std::to_string(s + 1) +
                                      (st.label.empty() ? std::string()
                                                        : " ('" + st.label + "')"));
      ++counts[c];
    }
    if (!(st.eweight * eweightglob > 0.0)) continue;
    ++n_weighted;
    ++compositions[counts];
  }

  // Rank of the fitted intercepts' columns: Gram-Schmidt (twice) on the
  // distinct count vectors.
  std::vector<std::vector<double>> basis;
  for (const auto &kv : compositions) {
    if (basis.size() == fitted.size()) break;
    std::vector<double> v;
    for (std::size_t k : fitted) v.push_back(static_cast<double>(kv.first[k]));
    double norm0 = 0.0;
    for (double x : v) norm0 += x * x;
    norm0 = std::sqrt(norm0);
    if (norm0 == 0.0) continue;
    for (int pass = 0; pass < 2; ++pass)
      for (const auto &b : basis) {
        double d = 0.0;
        for (std::size_t i = 0; i < v.size(); ++i) d += v[i] * b[i];
        for (std::size_t i = 0; i < v.size(); ++i) v[i] -= d * b[i];
      }
    double r = 0.0;
    for (double x : v) r += x * x;
    r = std::sqrt(r);
    if (r > 1e-10 * norm0) {
      for (double &x : v) x /= r;
      basis.push_back(v);
    }
  }
  if (basis.size() == fitted.size()) return;

  // Refuse, describing the data.
  std::vector<std::string> species;
  for (std::size_t k = 0; k < n_int; ++k)
    species.push_back(ctx.get<std::string>("ATOM", k));
  auto composition = [&](const std::vector<std::size_t> &counts) {
    std::string s;
    for (std::size_t k = 0; k < n_int; ++k)
      s += (k ? " " : "") + species[k] + std::to_string(counts[k]);
    return s;
  };
  std::string pinned_note;
  if (fitted.size() < n_int)
    pinned_note = " (FIXINDEX pins the other " +
                  std::to_string(n_int - fitted.size()) + ")";
  const std::string fitted_n =
      plural(fitted.size(), "intercept") +
      (fitted.size() == 1 ? " is fitted" : " are fitted") + pinned_note;
  std::string data;
  if (n_weighted == 0) {
    data = "no training structure has an energy weight (eweight x EWEIGHT > "
           "0), while " + fitted_n;
  } else {
    data = "the species counts of the " +
           plural(n_weighted, "energy-weighted structure") +
           " (eweight x EWEIGHT > 0) have rank " +
           std::to_string(basis.size()) + ", while " + fitted_n + ": ";
    if (compositions.size() == 1) {
      data += (n_weighted == 1 ? "its composition is "
                               : "every one of them has the composition ") +
              composition(compositions.begin()->first);
    } else {
      data += "their compositions are ";
      std::size_t shown = 0;
      for (const auto &kv : compositions) {
        if (shown == 4) {
          data += ", ...";
          break;
        }
        data += (shown++ ? ", " : "") + composition(kv.first) + " (" +
                plural(kv.second, "structure") + ")";
      }
    }
  }
  const std::string atoms = atom_list(ctx);
  throw std::runtime_error(
      "BIAS true: the training data cannot determine the per-species energy "
      "constants (the intercepts) of ATOM " + atoms + ". An intercept is "
      "fitted to energies only, and " + data + ". A fit would split the "
      "energy between the species arbitrarily (or overflow), and the "
      "per-atom energies, and the energy of any other composition, would be "
      "meaningless. Use BIAS false (ESHIFT_ATOM, ESHIFT_DBATOM or ESHIFT "
      "remove a per-species reference energy), add energy-weighted "
      "structures of other compositions (for example of the pure elements, "
      "or an isolated atom of each species), or pin intercepts with "
      "FIXINDEX/FIXWEIGHT (weights 1 to " + std::to_string(n_int) +
      " are the intercepts of " + atoms + ", in ATOM order).");
}

void require_descriptor_layout(const tadah::models::DC_Selector &dcs,
                               const tadah::core::Context &ctx,
                               const std::string &where) {
  const std::size_t dsize = ctx.get<std::size_t>("DSIZE");
  if (dcs.descriptor_size() != dsize)
    throw std::logic_error(
        where + ": the descriptor of DescriptorsCalc has DSIZE " +
        std::to_string(dsize) + " entries, but DC_Selector, the descriptor "
        "of pair_style tadah, has " + std::to_string(dcs.descriptor_size()) +
        " (" + std::to_string(dcs.bias) + " intercept columns): the two "
        "descriptor builders disagree.");
}

} // namespace mlip
} // namespace tadah
