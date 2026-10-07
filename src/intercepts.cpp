#include <tadah/mlip/intercepts.h>

#include <tadah/core/core_types.h>
#include <tadah/core/intercept_layout.h>
#include <tadah/core/registry.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/models/basis.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/models/functions/kernels/kern_base.h>
#include <tadah/models/functions/weights_fit.h>

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
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

/* The value of each intercept column cols[j] of f for an atom of each
 * species s, v[s][j]: f's energy of the unit weight cols[j] at the
 * descriptor whose only non-zero component is intercept component s (an
 * intercept column depends on an atom only through its species). */
std::vector<std::vector<double>>
intercept_values(const tadah::core::Context &ctx,
                 const tadah::models::Function_Base &f,
                 const std::vector<std::size_t> &cols) {
  const std::size_t n_int = tadah::core::n_intercept_columns(ctx);
  const std::size_t dsize = ctx.get<std::size_t>("DSIZE");
  if (dsize < n_int)
    throw std::logic_error(
        "the intercepts of " + f.get_label() + ": DSIZE is " +
        std::to_string(dsize) + ", fewer than the " +
        std::to_string(n_int) + " intercept columns (DescriptorsCalc sets "
        "DSIZE)");
  // A kernel evaluates its basis vectors against the descriptor: they must
  // have its length (a potential's of another descriptor read past it).
  const std::size_t bvs = f.basis_vector_size();
  if (bvs != 0 && bvs != dsize)
    throw std::logic_error(
        "the intercepts of " + f.get_label() + ": its basis vectors have " +
        std::to_string(bvs) + " values, but the descriptor has DSIZE " +
        std::to_string(dsize) + " entries (a kernel's basis vectors have the "
        "descriptor's length)");
  const std::size_t nw = f.weights_count(dsize);
  std::vector<std::vector<double>> v(n_int,
                                     std::vector<double>(cols.size(), 0.0));
  tadah::core::aed_type aed(dsize);
  tadah::core::t_type w(nw);
  for (std::size_t s = 0; s < n_int; ++s) {
    for (std::size_t k = 0; k < dsize; ++k) aed(k) = k == s ? 1.0 : 0.0;
    for (std::size_t j = 0; j < cols.size(); ++j) {
      for (std::size_t k = 0; k < nw; ++k) w(k) = k == cols[j] ? 1.0 : 0.0;
      v[s][j] = f.epredict(w, aed);
    }
  }
  return v;
}

/* The function whose intercept columns a fit of ctx has: f, but for a model
 * whose basis vectors the fit draws from the training data
 * (basis_drawn_from_training_data: M_KRR with a non-linear kernel). Every
 * basis such a fit draws starts with the intercept vectors e_0 .. e_{n0-1}
 * (Basis), and only columns k < N can be a kernel's intercepts
 * (Kern_Base::dot_product_intercept_columns), so the fit's intercepts are
 * those of a kernel built as f is, from ctx (its MPARAMS), with those
 * vectors alone as its basis, whatever other basis f holds: none before the
 * fit (a Trainer reads no BASIS), or a potential's of another fit. `hold`
 * keeps that kernel. */
const tadah::models::Function_Base &
fit_function(const tadah::core::Context &ctx,
             const tadah::models::Function_Base &f,
             std::unique_ptr<DM_Function_Base> &hold) {
  if (!basis_drawn_from_training_data(ctx)) return f;
  const std::size_t n0 = n_intercept_basis_vectors(ctx);
  const std::size_t dsize = ctx.get<std::size_t>("DSIZE");
  // A basis of f's that starts with those vectors (as M_KRR's after its
  // draw) gives the same intercepts: f is that kernel already.
  if (const auto *kf = dynamic_cast<const tadah::models::Kern_Base *>(&f)) {
    const tadah::core::Matrix &b = kf->get_basis();
    bool drawn = b.rows() == dsize && b.cols() >= n0;
    for (std::size_t k = 0; drawn && k < n0; ++k)
      for (std::size_t j = 0; drawn && j < dsize; ++j)
        if (b(j, k) != (j == k ? 1.0 : 0.0)) drawn = false;
    if (drawn) return f;
  }
  tadah::core::Context c = ctx;
  if (c.has_stored("BASIS")) c.remove("BASIS");   // the kernel would read it
  hold.reset(tadah::core::factory<DM_Function_Base,
                                  const tadah::core::Context &>(f.get_label(),
                                                                c));
  auto *k = dynamic_cast<tadah::models::Kern_Base *>(hold.get());
  if (k == nullptr)
    throw std::logic_error(
        "the intercepts of a fit of M_KRR " + ctx.get<std::string>("MODEL", 1) +
        ": the function " + f.get_label() + " is not a kernel");
  k->set_basis(intercept_basis_vectors(n0, dsize));
  return *hold;
}

/* "1 (Nb) and 6 (Ta)": the columns as weights, FIXINDEX numbering, each
 * labelled with its species when it is non-zero for that species alone. */
std::string weights_text(const tadah::core::Context &ctx,
                         const std::vector<std::size_t> &cols,
                         const std::vector<std::vector<double>> &v) {
  std::string s;
  for (std::size_t j = 0; j < cols.size(); ++j) {
    if (j) s += j + 1 == cols.size() ? " and " : ", ";
    s += std::to_string(cols[j] + 1);
    std::size_t nonzero = 0, species = 0;
    for (std::size_t k = 0; k < v.size(); ++k)
      if (v[k][j] != 0.0) {
        ++nonzero;
        species = k;
      }
    if (nonzero == 1) s += " (" + ctx.get<std::string>("ATOM", species) + ")";
  }
  return s;
}

}  // namespace

std::string intercept_weights_text(const tadah::core::Context &ctx,
                                   const tadah::models::Function_Base &f) {
  const std::vector<std::size_t> cols = f.intercept_columns(ctx);
  return weights_text(ctx, cols, intercept_values(ctx, f, cols));
}

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

std::string fixindex_intercept_warning(const tadah::core::Context &ctx,
                                       const tadah::models::Function_Base &fn) {
  if (!ctx.has_stored("FIXINDEX")) return "";
  const std::size_t n = tadah::core::n_intercept_columns(ctx);
  if (n < 2) return "";
  const std::string N = std::to_string(n);
  const std::string head = "FIXINDEX with BIAS true and " + N +
                           " ATOM species (" + atom_list(ctx) + "): ";
  const std::string before =
      "Before 1.6.0 BIAS true had one intercept shared by all species";
  // the weights the fit gives the function (a kernel's: of the basis it draws)
  std::unique_ptr<DM_Function_Base> hold;
  const tadah::models::Function_Base &f = fit_function(ctx, fn, hold);
  if (tadah::models::is_linear_function(f))
    return head + "weights 1 to " + N +
           " are the per-species intercepts, in ATOM order, and the "
           "descriptor weights start at index " + std::to_string(n + 1) +
           ". " + before + " and the descriptor weights started at index 2: "
           "an index written for that layout now pins another weight "
           "(descriptor weight k moved from index k + 1 to k + " + N + ").";
  std::string intercepts;
  const std::vector<std::size_t> cols = f.intercept_columns(ctx);
  if (!cols.empty())
    intercepts = "; its intercepts are weights " +
                 weights_text(ctx, cols, intercept_values(ctx, f, cols));
  else if (dynamic_cast<const tadah::models::Kern_Base *>(&f))
    intercepts = "; weights 1 to " + N + " belong to its intercept basis "
                 "vectors e_0 .. e_" + std::to_string(n - 1) +
                 ", one per species";
  return head + "the weights of " + f.get_label() + " are laid out for a "
         "descriptor with one intercept column per species" + intercepts +
         ". " + before + ": an index written for that layout may now pin "
         "another weight.";
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
                                   const tadah::core::Context &ctx,
                                   const tadah::models::Function_Base &fn) {
  const std::size_t n_int = tadah::core::n_intercept_columns(ctx);
  if (n_int == 0) return;
  const auto col = tadah::core::intercept_columns_by_Z(ctx);

  // The distinct species-count vectors of the energy-weighted structures
  // (an atom of an element not in ATOM is refused).
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

  // M_KRR implementation 2 solves (K + LAMBDA I) w = T on its basis vectors
  // (the intercept vectors with the target 0, the structures' mean AEDs with
  // their energies per atom), not the least squares of the structures'
  // energies: nothing to refuse.
  if (basis_drawn_from_training_data(ctx) && krr_implementation(ctx) == 2)
    return;

  // The intercept columns of the function as the fit has it (fit_function);
  // those FIXINDEX/FIXWEIGHT pin are not fitted.
  std::unique_ptr<DM_Function_Base> hold;
  const tadah::models::Function_Base &f = fit_function(ctx, fn, hold);
  const std::vector<std::size_t> cols = f.intercept_columns(ctx);
  if (cols.empty()) return;
  // The weights FIXINDEX pins, read as the fit reads them
  // (core::parse_fixindex; the fit checks them against its weight count).
  std::vector<bool> pinned_weight;
  for (std::size_t i : tadah::core::parse_fixindex(ctx, std::nullopt)) {
    if (pinned_weight.size() < i) pinned_weight.resize(i, false);
    pinned_weight[i - 1] = true;
  }
  std::vector<std::size_t> fitted;
  for (std::size_t c : cols)
    if (c >= pinned_weight.size() || !pinned_weight[c]) fitted.push_back(c);
  if (fitted.empty()) return;
  const std::vector<std::vector<double>> v = intercept_values(ctx, f, fitted);

  // Rank of the energy rows of the fitted intercept columns (the value of
  // column j of a structure: sum over species of count x v[s][j]):
  // Gram-Schmidt (twice) on the rows of the distinct compositions.
  std::vector<std::vector<double>> basis;
  for (const auto &kv : compositions) {
    if (basis.size() == fitted.size()) break;
    std::vector<double> row(fitted.size(), 0.0);
    for (std::size_t j = 0; j < fitted.size(); ++j)
      for (std::size_t k = 0; k < n_int; ++k)
        row[j] += static_cast<double>(kv.first[k]) * v[k][j];
    double norm0 = 0.0;
    for (double x : row) norm0 += x * x;
    norm0 = std::sqrt(norm0);
    if (norm0 == 0.0) continue;
    for (int pass = 0; pass < 2; ++pass)
      for (const auto &b : basis) {
        double d = 0.0;
        for (std::size_t i = 0; i < row.size(); ++i) d += row[i] * b[i];
        for (std::size_t i = 0; i < row.size(); ++i) row[i] -= d * b[i];
      }
    double r = 0.0;
    for (double x : row) r += x * x;
    r = std::sqrt(r);
    if (r > 1e-10 * norm0) {
      for (double &x : row) x /= r;
      basis.push_back(row);
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
  if (fitted.size() < cols.size())
    pinned_note = " (FIXINDEX pins the other " +
                  std::to_string(cols.size() - fitted.size()) + ")";
  const std::string fitted_n =
      plural(fitted.size(), "intercept") +
      (fitted.size() == 1 ? " is fitted" : " are fitted") + pinned_note;
  std::string data;
  if (n_weighted == 0) {
    data = "no training structure has an energy weight (eweight x EWEIGHT > "
           "0), while " + fitted_n;
  } else {
    data = "over the " + plural(n_weighted, "energy-weighted structure") +
           " (eweight x EWEIGHT > 0) the fitted intercept columns have "
           "rank " + std::to_string(basis.size()) + ", while " + fitted_n +
           ": ";
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
      "FIXINDEX/FIXWEIGHT (the intercepts of " + f.get_label() + " are "
      "weights " + weights_text(ctx, cols, intercept_values(ctx, f, cols)) +
      ").");
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
