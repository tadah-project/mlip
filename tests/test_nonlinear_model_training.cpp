// test_nonlinear_model_training.cpp
//
// Training and prediction of the non-linear models: M_KRR with Kern_RBF,
// Kern_Polynomial, Kern_Quadratic, Kern_LQ and Kern_Sigmoid, and M_BLR with
// BF_Polynomial2, through the Trainer that `tadah train` and `tadah hpo`
// use, against an independent reference written here: the closed-form
// features (kernel values, products of descriptor components) and their
// gradients, evaluated on the library's descriptors and descriptor
// derivatives, assembled into energy, force and stress rows, and solved
// by Householder QR in long double.
//
// Before the fix every one of these models was refused by
// DesignMatrix::build ("ColumnLayout: total_cols=3 disagrees with
// design-matrix width=5"): it compared the descriptor's column layout
// (DSIZE columns) with the design-matrix width, which only a linear
// function's equals (MLIP ffb2a13, 1.3.0-beta.1). The cases name the
// defect each would have met after that check alone: the EKM rotation of
// M_KRR_Train (columns appended on every training, a basis vector with
// K(b, b) = 0 filtered out of the caller's basis with an out-of-bounds
// write, rotated column 0 left out of the ridge penalty), the basis drawn
// with std::random_device, SBASIS 0 (a write into a basis of 0 columns).
// EKM::configure ran before the check: with Kern_Sigmoid here (c = -1,
// so K(e0, e0) = tanh(-0.5) < 0) it wrote past a matrix before the check
// threw, and some cases with a Kern_Sigmoid section aborted there
// ("malloc(): invalid size (unsorted)") instead of failing.
//
// Data: six Ta cells of 3 and 4 atoms, sheared and rattled, so that every
// atomic energy descriptor (AED) differs from every other (a 2-atom cell
// would not do: both atoms see the same distances); the cases REQUIRE it.
// D2_Blip with 3 functions (centres 2.8, 3.5, 4.2 A, width 2), WATOM 1,
// Cut_Cos at 5 A, INITMB false (DSIZE 3, or 4 with BIAS true); FORCE and
// STRESS true. The targets are arbitrary smooth numbers, so the
// least-squares solution has residuals (a fit that interpolated would not
// tell least squares from any other solution).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/column_block.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_rbf.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/models/m_krr.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/core/registry.h>
#include <tadah/models/cut_all.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/descriptors/d2/d2_blip.h>
#include <tadah/models/descriptors/dm/dm_dummy.h>
#include <tadah/models/functions/kernels/kern_base.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::core::t_type;
using tadah::mlip::Atom;
using tadah::mlip::ColumnBlock;
using tadah::mlip::ColumnLayout;
using tadah::mlip::DesignMatrix;
using tadah::mlip::DescriptorsCalc;
using tadah::mlip::DM_Function_Base;
using tadah::mlip::NNFinder;
using tadah::mlip::Normaliser;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

using ld = long double;
using Rows = std::vector<std::vector<ld>>;

constexpr double kRcut = 5.0;

/* ---- the models ------------------------------------------------------ */

struct ModelSpec {
  std::string name;                  // DYNAMIC_SECTION name
  std::vector<std::string> model;    // MODEL tokens
  std::vector<std::string> mparams;  // MPARAMS tokens (may be empty)
  bool kernel;                       // M_KRR with a basis
};

/* MPARAMS chosen for the scale of these descriptors (components 0.2 ..
 * 2.1, b . x up to ~5): sigma of the order of the distance between AEDs
 * and from e0, and a Sigmoid argument gamma b . x + c away from
 * saturation. */
const std::vector<ModelSpec> &nonlinear_models() {
  static const std::vector<ModelSpec> m = {
      {"M_KRR Kern_RBF", {"M_KRR", "Kern_RBF", "1"}, {"1.0"}, true},
      {"M_KRR Kern_Polynomial", {"M_KRR", "Kern_Polynomial", "1"},
       {"3", "0.5", "1.0"}, true},
      {"M_KRR Kern_Quadratic", {"M_KRR", "Kern_Quadratic", "1"}, {}, true},
      {"M_KRR Kern_LQ", {"M_KRR", "Kern_LQ", "1"}, {}, true},
      {"M_KRR Kern_Sigmoid", {"M_KRR", "Kern_Sigmoid", "1"},
       {"0.5", "-1.0"}, true},
      {"M_BLR BF_Polynomial2", {"M_BLR", "BF_Polynomial2"}, {}, false},
  };
  return m;
}

const ModelSpec &model_named(const std::string &name) {
  for (const auto &m : nonlinear_models())
    if (m.name == name) return m;
  throw std::logic_error("no model " + name);
}

constexpr std::size_t kSbasis = 5;  // e0 + 4 drawn AEDs

/* The training configuration: D2_Blip (3 functions) on Ta, Cut_Cos 5 A,
 * FORCE and STRESS true, LAMBDA 0, OALGO 1, SBASIS kSbasis. */
Context make_context(const ModelSpec &m, bool bias = false) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 3);
  c.add("TYPE2B", 3);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "2.8 3.5 4.2");
  c.add("SGRID2B", "2.0 2.0 2.0");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", kRcut);
  c.add("RCUTMB", kRcut);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "true");
  c.add("STRESS", "true");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", bias ? "true" : "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  for (const auto &t : m.model) c.add("MODEL", t);
  for (const auto &t : m.mparams) c.add("MPARAMS", t);
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  if (m.kernel) c.add("SBASIS", kSbasis);
  return c;
}

void set_key(Context &c, const std::string &key, const std::string &value) {
  c.remove(key);
  c.add(key, value);
}

/* ---- the data -------------------------------------------------------- */

/* A cell with rows a*[(1,0,0), (s0,1,0), (s1,s2,1)] and atoms at the given
 * fractional coordinates. */
Structure make_cell(double a, std::array<double, 3> shear,
                    const std::vector<std::array<double, 3>> &frac) {
  Structure st;
  st.cell.load(a, 0.0, 0.0,
               shear[0] * a, a, 0.0,
               shear[1] * a, shear[2] * a, a);
  for (const auto &f : frac) {
    double r[3];
    for (int k = 0; k < 3; ++k)
      r[k] = f[0] * st.cell(0, k) + f[1] * st.cell(1, k) + f[2] * st.cell(2, k);
    st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0, 0, 0));
  }
  return st;
}

/* Fills energy, forces and stress with arbitrary smooth numbers of the
 * size of real ones (eV, eV/A, eV/A^3). */
void set_targets(Structure &st, std::size_t s) {
  const double ds = static_cast<double>(s);
  st.energy = -8.1 * static_cast<double>(st.natoms()) +
              0.37 * std::sin(1.3 * ds + 0.2);
  for (std::size_t a = 0; a < st.natoms(); ++a)
    for (int k = 0; k < 3; ++k)
      st.atoms[a].force[k] =
          0.25 * std::sin(0.7 * static_cast<double>(a) + 1.1 * k + 0.3 * ds);
  for (int x = 0; x < 3; ++x)
    for (int y = x; y < 3; ++y) {
      const double v = 0.02 * std::cos(0.9 * x + 1.7 * y + 0.5 * ds);
      st.stress(x, y) = v;
      st.stress(y, x) = v;
    }
}

/* Six sheared and rattled cells, 3 and 4 atoms (21 AEDs). */
StructureDB training_db() {
  std::vector<Structure> cells = {
      make_cell(3.30, {0.05, 0.02, -0.03},
                {{0.00, 0.01, 0.02}, {0.48, 0.52, 0.05}, {0.27, 0.55, 0.51}}),
      make_cell(3.45, {-0.04, 0.03, 0.06},
                {{0.01, -0.02, 0.00}, {0.52, 0.47, 0.03},
                 {0.49, 0.02, 0.53}, {-0.03, 0.51, 0.48}}),
      make_cell(3.15, {0.07, -0.05, 0.02},
                {{0.02, -0.01, 0.00}, {0.33, 0.31, 0.36}, {0.68, 0.64, 0.70}}),
      make_cell(3.60, {0.00, 0.08, -0.02},
                {{0.00, 0.00, 0.03}, {0.46, 0.50, 0.01},
                 {0.53, 0.04, 0.47}, {0.02, 0.55, 0.52}}),
      make_cell(3.22, {-0.06, 0.01, 0.04},
                {{0.01, 0.03, -0.02}, {0.51, 0.43, 0.10}, {0.22, 0.61, 0.57}}),
      make_cell(3.52, {0.03, -0.02, -0.05},
                {{-0.02, 0.01, 0.01}, {0.49, 0.53, -0.02},
                 {0.47, -0.03, 0.50}, {0.04, 0.48, 0.55}}),
  };
  StructureDB db;
  for (std::size_t s = 0; s < cells.size(); ++s) {
    set_targets(cells[s], s);
    db.add(cells[s]);
  }
  return db;
}

/* A cell the models are not trained on (the finite-difference checks). */
Structure held_out_cell() {
  return make_cell(3.38, {0.04, -0.03, 0.05},
                   {{0.03, -0.02, 0.01}, {0.45, 0.53, 0.08},
                    {0.31, 0.58, 0.49}});
}

/* ---- the reference --------------------------------------------------- */

/* The closed-form features of the design-matrix columns and their
 * gradients with respect to the descriptor x: for a kernel, column j is
 * K(b_j, x) with b_j the j-th basis vector; for BF_Polynomial2, the
 * column of (i, ii), i <= ii, is x_i x_ii. */
struct Reference {
  std::string fn;
  std::vector<std::vector<ld>> basis;  // basis vectors (kernels)
  std::vector<ld> mp;                  // MPARAMS
  std::size_t dim = 0;                 // descriptor length

  std::size_t cols() const {
    return fn == "BF_Polynomial2" ? dim * (dim + 1) / 2 : basis.size();
  }

  void eval(const tadah::core::aed_type &xd, std::vector<ld> &phi,
            std::vector<std::vector<ld>> &dphi) const {
    std::vector<ld> x(dim);
    for (std::size_t d = 0; d < dim; ++d) x[d] = xd[d];
    const std::size_t n = cols();
    phi.assign(n, 0);
    dphi.assign(n, std::vector<ld>(dim, 0));
    if (fn == "BF_Polynomial2") {
      std::size_t col = 0;
      for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t ii = i; ii < dim; ++ii, ++col) {
          phi[col] = x[i] * x[ii];
          dphi[col][i] += x[ii];
          dphi[col][ii] += x[i];
        }
      return;
    }
    for (std::size_t j = 0; j < n; ++j) {
      const std::vector<ld> &b = basis[j];
      ld bx = 0, r2 = 0;
      for (std::size_t d = 0; d < dim; ++d) {
        bx += b[d] * x[d];
        r2 += (x[d] - b[d]) * (x[d] - b[d]);
      }
      if (fn == "Kern_RBF") {
        const ld s = mp[0];
        const ld k = std::exp(-r2 / (2 * s * s));
        phi[j] = k;
        for (std::size_t d = 0; d < dim; ++d)
          dphi[j][d] = -(x[d] - b[d]) / (s * s) * k;
        continue;
      }
      ld f = 0, df = 0;  // K = f(b . x), dK/dx = df * b
      if (fn == "Kern_Polynomial") {
        const ld deg = mp[0], g = mp[1], c = mp[2];
        const ld u = g * bx + c;
        f = std::pow(u, deg);
        df = deg * g * std::pow(u, deg - 1);
      } else if (fn == "Kern_Quadratic") {
        f = bx * bx;
        df = 2 * bx;
      } else if (fn == "Kern_LQ") {
        f = bx * bx + bx;
        df = 2 * bx + 1;
      } else if (fn == "Kern_Sigmoid") {
        const ld g = mp[0], c = mp[1];
        f = std::tanh(g * bx + c);
        df = g * (1 - f * f);
      } else {
        throw std::logic_error("no reference for " + fn);
      }
      phi[j] = f;
      for (std::size_t d = 0; d < dim; ++d) dphi[j][d] = df * b[d];
    }
  }
};

/* The reference of a trained model: its function, MPARAMS and the basis
 * read back from the potential it writes (get_param_file: SBASIS, BASIS),
 * since the basis is drawn from the data. */
Reference reference_of(Trainer &tr, const ModelSpec &m) {
  Reference r;
  r.fn = m.model[1];
  for (const auto &t : m.mparams) r.mp.push_back(std::stold(t));
  r.dim = tr.context.get<std::size_t>("DSIZE");
  if (m.kernel) {
    const Context pot = tr.model->get_param_file();
    const std::size_t s = pot.get<std::size_t>("SBASIS");
    REQUIRE(s == tr.context.get<std::size_t>("SBASIS"));
    REQUIRE(pot.size("BASIS") == s * r.dim);
    r.basis.assign(s, std::vector<ld>(r.dim));
    for (std::size_t j = 0; j < s; ++j)
      for (std::size_t d = 0; d < r.dim; ++d)
        r.basis[j][d] = pot.get<double>("BASIS", j * r.dim + d);
  }
  return r;
}

/* Energy, force and stress rows of the structures (each with its neighbour
 * list) and their targets, in DesignMatrix's order: per structure the
 * energy, the forces atom by atom, then the stress xx xy xz yy yz zz. A
 * force row is -sum over the neighbours j of i of
 * [grad phi(x_i) . fd_ij(:,k) - grad phi(x_j) . fd_ji(:,k)]; a stress row
 * (1/V) sum_i sum_j 0.5 (r_i - r_j)_x [grad phi(x_i) . fd_ij(:,y) -
 * grad phi(x_j) . fd_ji(:,y)]. */
void build_rows(const Reference &ref, DescriptorsCalc<> &dc,
                const StructureDB &db, Rows &A, std::vector<ld> &T) {
  A.clear();
  T.clear();
  const std::size_t n = ref.cols();
  std::vector<ld> phi;
  std::vector<std::vector<ld>> dphi;
  for (std::size_t s = 0; s < db.size(); ++s) {
    const Structure &st = db(s);
    const StDescriptors sd = dc.calc(st);
    const std::size_t na = st.natoms();
    std::vector<std::vector<ld>> f(na);
    std::vector<std::vector<std::vector<ld>>> g(na);
    for (std::size_t a = 0; a < na; ++a) {
      ref.eval(sd.get_aed(a), phi, dphi);
      f[a] = phi;
      g[a] = dphi;
    }
    // grad phi_c(x_a) . fd(:, k)
    auto dot = [&](std::size_t a, std::size_t c,
                   const tadah::core::fd_type &fd, int k) {
      ld v = 0;
      for (std::size_t d = 0; d < ref.dim; ++d) v += g[a][c][d] * fd(d, k);
      return v;
    };
    std::vector<ld> row(n, 0);
    for (std::size_t a = 0; a < na; ++a)
      for (std::size_t c = 0; c < n; ++c) row[c] += f[a][c];
    A.push_back(row);
    T.push_back(st.energy);
    for (std::size_t i = 0; i < na; ++i)
      for (int k = 0; k < 3; ++k) {
        std::fill(row.begin(), row.end(), 0);
        for (std::size_t jj = 0; jj < sd.fd[i].size(); ++jj) {
          const std::size_t j = st.nn->idx[i][jj];
          const std::size_t ii = st.get_nn_iindex(i, j, jj);
          for (std::size_t c = 0; c < n; ++c)
            row[c] -= dot(i, c, sd.fd[i][jj], k) - dot(j, c, sd.fd[j][ii], k);
        }
        A.push_back(row);
        T.push_back(st.atoms[i].force[k]);
      }
    const ld vinv = 1.0L / st.get_volume();
    for (int x = 0; x < 3; ++x)
      for (int y = x; y < 3; ++y) {
        std::fill(row.begin(), row.end(), 0);
        for (std::size_t i = 0; i < na; ++i)
          for (std::size_t jj = 0; jj < sd.fd[i].size(); ++jj) {
            const std::size_t j = st.nn->idx[i][jj];
            const std::size_t ii = st.get_nn_iindex(i, j, jj);
            const ld rx = st.atoms[i].position[x] - st.nn_pos(i, jj)[x];
            for (std::size_t c = 0; c < n; ++c)
              row[c] += vinv * 0.5L * rx *
                        (dot(i, c, sd.fd[i][jj], y) - dot(j, c, sd.fd[j][ii], y));
          }
        A.push_back(row);
        T.push_back(st.stress(x, y));
      }
  }
}

/* min ||A w - T|| by Householder QR in long double, over the columns
 * `active` (the others get w = 0). rcond is min |R_kk| / max |R_kk|, a
 * proxy for the inverse condition number of the active columns. */
struct LsSolution {
  std::vector<ld> w;
  ld rcond = 0;
};
LsSolution least_squares(const Rows &A0, const std::vector<ld> &T0,
                         const std::vector<bool> &active) {
  const std::size_t m = A0.size();
  std::vector<std::size_t> cols;
  for (std::size_t c = 0; c < active.size(); ++c)
    if (active[c]) cols.push_back(c);
  const std::size_t n = cols.size();
  REQUIRE(m >= n);
  Rows A(m, std::vector<ld>(n));
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) A[i][j] = A0[i][cols[j]];
  std::vector<ld> b = T0;
  for (std::size_t k = 0; k < n; ++k) {
    ld norm = 0;
    for (std::size_t i = k; i < m; ++i) norm += A[i][k] * A[i][k];
    norm = std::sqrt(norm);
    if (norm == 0) continue;
    const ld alpha = A[k][k] > 0 ? -norm : norm;
    std::vector<ld> v(m - k);
    for (std::size_t i = k; i < m; ++i) v[i - k] = A[i][k];
    v[0] -= alpha;
    ld vv = 0;
    for (ld e : v) vv += e * e;
    for (std::size_t j = k; j < n; ++j) {
      ld s = 0;
      for (std::size_t i = k; i < m; ++i) s += v[i - k] * A[i][j];
      for (std::size_t i = k; i < m; ++i) A[i][j] -= 2 * s / vv * v[i - k];
    }
    ld s = 0;
    for (std::size_t i = k; i < m; ++i) s += v[i - k] * b[i];
    for (std::size_t i = k; i < m; ++i) b[i] -= 2 * s / vv * v[i - k];
  }
  LsSolution out;
  ld rmax = 0, rmin = -1;
  for (std::size_t k = 0; k < n; ++k) {
    const ld r = std::fabs(A[k][k]);
    rmax = std::max(rmax, r);
    rmin = rmin < 0 ? r : std::min(rmin, r);
  }
  out.rcond = n ? rmin / rmax : 1;
  std::vector<ld> wa(n, 0);
  for (std::size_t k = n; k-- > 0;) {
    ld s = b[k];
    for (std::size_t j = k + 1; j < n; ++j) s -= A[k][j] * wa[j];
    wa[k] = s / A[k][k];
  }
  out.w.assign(active.size(), 0);
  for (std::size_t j = 0; j < n; ++j) out.w[cols[j]] = wa[j];
  return out;
}

LsSolution least_squares(const Rows &A, const std::vector<ld> &T) {
  return least_squares(A, T, std::vector<bool>(A.empty() ? 0 : A[0].size(), true));
}

/* The ridge problem min ||A w - T||^2 + lambda sum_{j >= n_int} w_j^2 as
 * least squares with the rows sqrt(lambda) e_j appended. */
LsSolution ridge(Rows A, std::vector<ld> T, ld lambda, std::size_t n_int) {
  const std::size_t n = A.empty() ? 0 : A[0].size();
  for (std::size_t j = n_int; j < n; ++j) {
    std::vector<ld> row(n, 0);
    row[j] = std::sqrt(lambda);
    A.push_back(row);
    T.push_back(0);
  }
  return least_squares(A, T);
}

/* max_j |w_j - ref_j| / max_j |ref_j| */
double rel_diff(const t_type &w, const std::vector<ld> &ref) {
  REQUIRE(w.size() == ref.size());
  ld num = 0, den = 0;
  for (std::size_t j = 0; j < ref.size(); ++j) {
    num = std::max(num, std::fabs(ld(w(j)) - ref[j]));
    den = std::max(den, std::fabs(ref[j]));
  }
  return static_cast<double>(num / den);
}

std::string to_text(const t_type &w) {
  std::ostringstream o;
  o.precision(17);
  for (std::size_t j = 0; j < w.size(); ++j) o << w(j) << ' ';
  return o.str();
}

std::string to_text(const std::vector<ld> &w) {
  std::ostringstream o;
  o.precision(17);
  for (ld v : w) o << static_cast<double>(v) << ' ';
  return o.str();
}

/* Every AED of the data differs from every other: a precondition of a
 * full-rank kernel design matrix (two equal basis vectors give two equal
 * columns). */
void require_distinct_aeds(DescriptorsCalc<> &dc, const StructureDB &db) {
  std::vector<std::vector<double>> aeds;
  for (std::size_t s = 0; s < db.size(); ++s) {
    const StDescriptors sd = dc.calc(db(s));
    for (std::size_t a = 0; a < sd.naed(); ++a) {
      std::vector<double> v;
      for (std::size_t d = 0; d < sd.dim(); ++d) v.push_back(sd.get_aed(a)[d]);
      aeds.push_back(v);
    }
  }
  for (std::size_t p = 0; p < aeds.size(); ++p)
    for (std::size_t q = p + 1; q < aeds.size(); ++q) {
      double dmax = 0;
      for (std::size_t d = 0; d < aeds[p].size(); ++d)
        dmax = std::max(dmax, std::fabs(aeds[p][d] - aeds[q][d]));
      INFO("AEDs " << p << " and " << q);
      REQUIRE(dmax > 1e-3);
    }
}

/* A trained model with its data. */
struct Trained {
  MLIPWorkspaceManager ws;
  Context ctx;
  std::unique_ptr<Trainer> tr;
  StructureDB db;

  Trained(const Trained &) = delete;
  Trained &operator=(const Trained &) = delete;
  explicit Trained(Context c, StructureDB data = training_db())
      : ctx(std::move(c)), db(std::move(data)) {
    tr = std::make_unique<Trainer>(ctx, ws);
  }
  void train() { tr->train(db); }
};

/* Checks the trained model's weights against the reference least-squares
 * solution of its features, and its predictions against Phi w. */
void check_fit(Trained &t, const ModelSpec &m) {
  const Reference ref = reference_of(*t.tr, m);
  Rows A;
  std::vector<ld> T;
  build_rows(ref, t.tr->dc, t.db, A, T);
  const LsSolution sol = least_squares(A, T);
  INFO("reference rcond (min/max |R_kk|) " << static_cast<double>(sol.rcond));
  REQUIRE(sol.rcond > 1e-9);  // full rank, well enough conditioned
  const t_type w = t.tr->model->get_weights();
  INFO("weights   " << to_text(w));
  INFO("reference " << to_text(sol.w));
  REQUIRE(w.size() == ref.cols());
  CHECK(rel_diff(w, sol.w) < 1e-8);

  // The model's predictions (kernel/basis-function epredict and fpredict,
  // M_Tadah_Base's force and stress sums) equal the rows times w.
  Context pc = t.tr->context;
  const StructureDB pred = t.tr->model->predict(pc, t.db, t.tr->dc);
  std::size_t r = 0;
  for (std::size_t s = 0; s < t.db.size(); ++s) {
    auto row_dot = [&](std::size_t i) {
      ld v = 0;
      for (std::size_t c = 0; c < ref.cols(); ++c) v += A[i][c] * ld(w(c));
      return static_cast<double>(v);
    };
    INFO("structure " << s);
    CHECK(pred(s).energy == Approx(row_dot(r++)).epsilon(1e-12).margin(1e-10));
    for (std::size_t a = 0; a < t.db(s).natoms(); ++a)
      for (int k = 0; k < 3; ++k)
        CHECK(pred(s).atoms[a].force[k] ==
              Approx(row_dot(r++)).epsilon(1e-10).margin(1e-11));
    for (int x = 0; x < 3; ++x)
      for (int y = x; y < 3; ++y)
        CHECK(pred(s).stress(x, y) ==
              Approx(row_dot(r++)).epsilon(1e-10).margin(1e-12));
  }
  CHECK(r == A.size());
}

}  // namespace

/* Pins: each non-linear model trains through the Trainer with FORCE and
 * STRESS (no throw); a kernel's potential carries SBASIS 5 and 5 x DSIZE
 * BASIS values; the trained weights equal the least-squares solution of
 * the reference rows (closed-form features of the read-back basis on the
 * library's AEDs and descriptor derivatives, Householder QR in long
 * double; the reference is full rank, rcond > 1e-9) to 1e-8 relative
 * (measured when the test was written: 1.3e-12 for Kern_Quadratic, whose
 * reference rcond is 1e-3, and below 2e-14 for the others); and the
 * model's predicted energies, forces and stresses equal the reference
 * rows times the weights (1e-12 / 1e-10 relative).
 * Before the fix: tr.train() threw "ColumnLayout: total_cols=3 disagrees
 * with design-matrix width=5" for every kernel (width 6 for
 * BF_Polynomial2). */
TEST_CASE("Non-linear models train through the design matrix and fit the "
          "least-squares solution of their features",
          "[nonlinear][train]") {
  PeriodicTable::initialize();
  for (const ModelSpec &m : nonlinear_models()) {
    DYNAMIC_SECTION(m.name) {
      Trained t(make_context(m));
      REQUIRE_NOTHROW(t.train());
      require_distinct_aeds(t.tr->dc, t.db);
      check_fit(t, m);
    }
  }
}

/* Pins: BF_Polynomial2 fitted to energies, forces and stresses generated
 * by its own features with the weights w*_c = cos(1.3 c + 0.4) (c = 0 ..
 * 5) recovers w* to 1e-9 relative (the 6 columns have full rank).
 * Before the fix: tr.train() threw "ColumnLayout: total_cols=3 disagrees
 * with design-matrix width=6". */
TEST_CASE("BF_Polynomial2 recovers the weights that generated its data",
          "[nonlinear][train]") {
  PeriodicTable::initialize();
  const ModelSpec &m = model_named("M_BLR BF_Polynomial2");
  Trained t(make_context(m));
  // The rows do not depend on the targets: compute them on the data, set
  // the targets to Phi w*, train.
  t.tr->nnf.calc(t.db, t.tr->context);
  Reference ref;
  ref.fn = "BF_Polynomial2";
  ref.dim = t.tr->context.get<std::size_t>("DSIZE");
  REQUIRE(ref.cols() == 6);
  Rows A;
  std::vector<ld> T;
  build_rows(ref, t.tr->dc, t.db, A, T);
  std::vector<ld> wstar(ref.cols());
  for (std::size_t c = 0; c < wstar.size(); ++c)
    wstar[c] = std::cos(1.3L * c + 0.4L);
  std::vector<ld> y(A.size(), 0);
  for (std::size_t i = 0; i < A.size(); ++i)
    for (std::size_t c = 0; c < wstar.size(); ++c) y[i] += A[i][c] * wstar[c];
  std::size_t r = 0;
  for (std::size_t s = 0; s < t.db.size(); ++s) {
    Structure &st = t.db.structures[s];
    st.energy = static_cast<double>(y[r++]);
    for (std::size_t a = 0; a < st.natoms(); ++a)
      for (int k = 0; k < 3; ++k) st.atoms[a].force[k] = static_cast<double>(y[r++]);
    for (int x = 0; x < 3; ++x)
      for (int yy = x; yy < 3; ++yy) {
        st.stress(x, yy) = static_cast<double>(y[r]);
        st.stress(yy, x) = static_cast<double>(y[r++]);
      }
  }
  REQUIRE(least_squares(A, y).rcond > 1e-9);
  REQUIRE_NOTHROW(t.train());
  const t_type w = t.tr->model->get_weights();
  INFO("weights " << to_text(w) << "\nw*      " << to_text(wstar));
  CHECK(rel_diff(w, wstar) < 1e-9);
}

namespace {

/* st deformed by F = I + e with the symmetric strain e_ij = e_ji = d/2
 * (e_ii = d): every lattice vector and every position goes to F v. */
Structure strained(const Structure &st, int i, int j, double d) {
  double F[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  F[i][j] += 0.5 * d;
  F[j][i] += 0.5 * d;
  auto map = [&F](double x, double y, double z, int c) {
    return F[c][0] * x + F[c][1] * y + F[c][2] * z;
  };
  Structure out = st;
  for (int r = 0; r < 3; ++r) {
    const double x = st.cell(r, 0), y = st.cell(r, 1), z = st.cell(r, 2);
    for (int c = 0; c < 3; ++c) out.cell(r, c) = map(x, y, z, c);
  }
  for (std::size_t a = 0; a < st.natoms(); ++a) {
    const auto &p = st.atoms[a].position;
    for (int c = 0; c < 3; ++c)
      out.atoms[a].position[c] = map(p[0], p[1], p[2], c);
  }
  return out;
}

double energy_of(Trainer &tr, Structure st) {
  NNFinder().calc(st, kRcut);
  return tr.model->epredict(tr.dc.calc(st));
}

}  // namespace

/* Pins: on a held-out cell, each trained non-linear model's predicted
 * forces equal -dE/dr and its predicted stress (1/V) dE/d(eps) of its own
 * predicted energy, by central differences (h = 1e-5 A on every atom
 * coordinate; h = 1e-5 on each symmetric strain, cell and atoms moved
 * together; the neighbour list rebuilt each time), to 1e-6 eV/A and
 * 1e-7 eV/A^3 (measured when the test was written: at most 5.3e-9 eV/A
 * and 2.4e-9 eV/A^3); the forces and stresses checked are not all small
 * (max |F| > 0.5 eV/A, max |sigma| > 0.05 eV/A^3; measured 0.98 .. 4.2
 * and 0.13 .. 0.87).
 * Before the fix: training threw (the ColumnLayout check), so no model
 * was there to differentiate. */
TEST_CASE("Non-linear models: forces and stress are derivatives of the "
          "predicted energy",
          "[nonlinear][train][derivatives]") {
  PeriodicTable::initialize();
  const double h = 1e-5;
  for (const ModelSpec &m : nonlinear_models()) {
    DYNAMIC_SECTION(m.name) {
      Trained t(make_context(m));
      REQUIRE_NOTHROW(t.train());
      Trainer &tr = *t.tr;
      Structure st0 = held_out_cell();
      NNFinder().calc(st0, kRcut);
      Context pc = tr.context;
      const Structure p = tr.model->predict(pc, st0, tr.dc);
      double fmax = 0, smax = 0;
      for (std::size_t a = 0; a < st0.natoms(); ++a)
        for (int k = 0; k < 3; ++k) {
          Structure sp = held_out_cell(), sm = held_out_cell();
          sp.atoms[a].position[k] += h;
          sm.atoms[a].position[k] -= h;
          const double fd = -(energy_of(tr, sp) - energy_of(tr, sm)) / (2 * h);
          INFO("atom " << a << " axis " << k << ": predicted "
                       << p.atoms[a].force[k] << ", -dE/dr " << fd);
          CHECK(p.atoms[a].force[k] == Approx(fd).margin(1e-6));
          fmax = std::max(fmax, std::fabs(fd));
        }
      const double V = st0.get_volume();
      for (int i = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j) {
          const double ep = energy_of(tr, strained(held_out_cell(), i, j, h));
          const double em = energy_of(tr, strained(held_out_cell(), i, j, -h));
          const double fd = (ep - em) / (2 * h * V);
          INFO("sigma(" << i << "," << j << "): predicted " << p.stress(i, j)
                        << ", (1/V) dE/d(eps) " << fd);
          CHECK(p.stress(i, j) == Approx(fd).margin(1e-7));
          smax = std::max(smax, std::fabs(fd));
        }
      CHECK(fmax > 0.5);
      CHECK(smax > 0.05);
    }
  }
}

/* Pins: the column layout DesignMatrix::build records is the function's:
 * for each kernel one block labelled by the kernel, family "kernel",
 * offset 0, SBASIS (5) columns; for BF_Polynomial2 one block
 * "BF_Polynomial2", family "BF", D(D+1)/2 = 6 columns; for BF_Linear and
 * Kern_Linear the descriptor's blocks as before: "bias" (1 column, with
 * BIAS true) then "D2_Blip" (3 columns).
 * Before the fix: the build threw for every non-linear function (the
 * ColumnLayout check; tr.train() threw first); the linear cases passed (a
 * pin). */
TEST_CASE("The design-matrix column layout is the function's",
          "[nonlinear][layout]") {
  PeriodicTable::initialize();
  for (const ModelSpec &m : nonlinear_models()) {
    DYNAMIC_SECTION(m.name) {
      Trained t(make_context(m));
      REQUIRE_NOTHROW(t.train());  // a kernel draws its basis
      DesignMatrix<DM_Function_Base &> dm(*t.tr->fb, t.tr->context);
      Normaliser norm(t.tr->context);
      REQUIRE_NOTHROW(dm.build(t.db, norm, t.tr->dc, false));
      const ColumnLayout &L = dm.layout();
      REQUIRE(L.blocks.size() == 1);
      const ColumnBlock &b = L.blocks[0];
      CHECK(b.descriptor_label == m.model[1]);
      CHECK(b.family == (m.kernel ? "kernel" : "BF"));
      CHECK(b.col_offset == 0);
      CHECK(b.col_count == (m.kernel ? kSbasis : 6));
    }
  }
  const std::vector<ModelSpec> linear = {
      {"M_BLR BF_Linear", {"M_BLR", "BF_Linear"}, {}, false},
      {"M_KRR Kern_Linear", {"M_KRR", "Kern_Linear", "1"}, {}, false}};
  for (const ModelSpec &m : linear) {
    DYNAMIC_SECTION(m.name << " (a pin)") {
      Trained t(make_context(m, /*bias=*/true));
      REQUIRE_NOTHROW(t.train());
      DesignMatrix<DM_Function_Base &> dm(*t.tr->fb, t.tr->context);
      Normaliser norm(t.tr->context);
      REQUIRE_NOTHROW(dm.build(t.db, norm, t.tr->dc, false));
      const ColumnLayout &L = dm.layout();
      REQUIRE(L.blocks.size() == 2);
      CHECK(L.blocks[0].descriptor_label == "bias");
      CHECK(L.blocks[0].family == "bias");
      CHECK(L.blocks[0].col_count == 1);
      CHECK(L.blocks[1].descriptor_label == "D2_Blip");
      CHECK(L.blocks[1].family == "D2");
      CHECK(L.blocks[1].col_offset == 1);
      CHECK(L.blocks[1].col_count == 3);
    }
  }
}

namespace {

/* DescriptorsCalc whose layout reports one column more than the
 * descriptor has: a stale descriptor layout. */
struct ExtraColumnDC : DescriptorsCalc<> {
  using DescriptorsCalc<>::DescriptorsCalc;
  ColumnLayout layout(const Context &c) const override {
    ColumnLayout L = DescriptorsCalc<>::layout(c);
    ColumnBlock extra;
    extra.descriptor_label = "extra";
    extra.family = "DM";
    extra.col_offset = L.total_cols();
    extra.col_count = 1;
    L.blocks.push_back(extra);
    return L;
  }
};

}  // namespace

/* Pins: a descriptor layout of DSIZE + 1 columns (4 for D2_Blip 3) is
 * refused by DesignMatrix::build with a message naming DSIZE ("disagrees
 * with DSIZE=3"), for a linear function (BF_Linear) and a non-linear one
 * (Kern_RBF with the basis of a training): the guard against a stale
 * descriptor layout holds for every function.
 * Before the fix: the build threw for both, but compared the layout with
 * the design-matrix width ("disagrees with design-matrix width=3" for
 * BF_Linear; the Kern_RBF model could not be trained, its training threw
 * the same check), so the DSIZE CHECKs failed. */
TEST_CASE("DesignMatrix refuses a descriptor layout that disagrees with "
          "DSIZE",
          "[nonlinear][layout]") {
  PeriodicTable::initialize();
  const std::vector<ModelSpec> models = {
      {"M_BLR BF_Linear", {"M_BLR", "BF_Linear"}, {}, false},
      model_named("M_KRR Kern_RBF")};
  for (const ModelSpec &m : models) {
    DYNAMIC_SECTION(m.name) {
      Trained t(make_context(m));
      REQUIRE_NOTHROW(t.train());
      Trainer &tr = *t.tr;
      ExtraColumnDC bad(tr.context, *tr.DCS.d2b, *tr.DCS.dmb, *tr.DCS.c2b,
                        *tr.DCS.cmb);
      DesignMatrix<DM_Function_Base &> dm(*tr.fb, tr.context);
      Normaliser norm(tr.context);
      std::string msg;
      try {
        dm.build(t.db, norm, bad, false);
      } catch (const std::exception &e) {
        msg = e.what();
      }
      INFO("message: " << msg);
      CHECK(msg.find("disagrees with DSIZE=3") != std::string::npos);
      CHECK(msg.find("total_cols=4") != std::string::npos);
    }
  }
}

/* Pins: one Trainer trained, its model reset (reset_trained()) and its
 * descriptors updated in place (DC_Selector::reinit) and trained again,
 * three times, with SGRID2B changed between the fits (2.0, 1.8, 2.2 for
 * every function: new descriptors, so a new basis): every fit has SBASIS
 * (5) weights for a kernel (6 for BF_Polynomial2) and equals the reference
 * least-squares solution of its own descriptors and basis to 1e-8
 * relative. (tadah hpo reused one Trainer so, through Trainer::reinit,
 * until it built a Trainer per evaluation; the model and descriptor calls
 * stay public.)
 * Before the fix: passes as well (a pin of the EKM fix of 1.6.0, before
 * which the first fit threw (the ColumnLayout check); with that check
 * alone fixed, every kernel's second fit wrote SBASIS 10 for 5 basis
 * vectors: EKM::configure appended the indices of every training to a
 * member it never cleared, and replaced M_KRR's basis by a 10-column one;
 * with Kern_Sigmoid the process aborted ("malloc(): invalid size")). It
 * called Trainer::reinit, which this change removes. */
TEST_CASE("Retraining one Trainer's model refits every non-linear model",
          "[nonlinear][train][retrain]") {
  PeriodicTable::initialize();
  for (const ModelSpec &m : nonlinear_models()) {
    DYNAMIC_SECTION(m.name) {
      Trained t(make_context(m));
      for (const char *sgrid : {"2.0 2.0 2.0", "1.8 1.8 1.8", "2.2 2.2 2.2"}) {
        INFO("SGRID2B " << sgrid);
        if (t.tr->model->is_trained()) {
          set_key(t.tr->context, "SGRID2B", sgrid);
          t.tr->model->reset_trained();
          REQUIRE_FALSE(t.tr->DCS.reinit(t.tr->context));  // same size
        }
        REQUIRE_NOTHROW(t.train());
        check_fit(t, m);
      }
    }
  }
}

namespace {

/* A 5-atom cell (five distinct AEDs) and two structures of one isolated
 * atom each (no neighbour within the cutoff in a 20 A cube): two zero
 * AEDs, seven in all. Kern_Quadratic's columns (b . x)^2 are quadratic
 * forms in the 3 descriptor components, which span 6 dimensions: e0 and
 * at most five non-zero AEDs keep the non-zero columns independent. */
StructureDB db_with_isolated_atoms() {
  StructureDB db;
  Structure five = make_cell(3.9, {0.06, -0.04, 0.03},
                             {{0.01, 0.02, -0.01}, {0.47, 0.52, 0.04},
                              {0.53, 0.01, 0.49}, {0.02, 0.49, 0.55},
                              {0.26, 0.24, 0.27}});
  set_targets(five, 0);
  db.add(five);
  for (int k = 0; k < 2; ++k) {
    Structure st = make_cell(20.0, {0, 0, 0}, {{0.1 + 0.3 * k, 0.2, 0.3}});
    set_targets(st, 1 + k);
    db.add(st);
  }
  return db;
}

}  // namespace

/* Pins: with two isolated atoms in the data (two AEDs equal to 0, of
 * seven) and SBASIS 7 (the draw takes 6 AEDs, so at least one zero AED is
 * a basis vector, and K(0, x) = 0 for Kern_Quadratic), M_KRR
 * Kern_Quadratic trains; the basis keeps all 7 vectors (BASIS has 7 x 3
 * values) and contains a zero vector; the weight of every zero basis
 * vector is 0 (|w| < 1e-12 max|w|: its column is 0, and the minimum-norm
 * solution of GELSD gives it nothing); the other weights equal the
 * reference least-squares solution over the non-zero columns (full rank,
 * rcond > 1e-9) to 1e-8 relative.
 * Before the fix: training threw (the ColumnLayout check). With that check
 * alone fixed (seen in a scratch build), EKM::configure dropped both zero
 * vectors from M_KRR's basis (writing the others into a matrix too narrow
 * for their indices) while the design matrix kept them: the potential had
 * SBASIS 5 for 7 weights, and the process then died (SIGSEGV). */
TEST_CASE("A basis vector with K(b, b) = 0 is an ordinary column",
          "[nonlinear][train][zero_aed]") {
  PeriodicTable::initialize();
  const ModelSpec &m = model_named("M_KRR Kern_Quadratic");
  Context c = make_context(m);
  const std::size_t n_aed = 5 + 2;
  set_key(c, "SBASIS", std::to_string(n_aed));
  Trained t(c, db_with_isolated_atoms());
  REQUIRE_NOTHROW(t.train());
  const Reference ref = reference_of(*t.tr, m);
  REQUIRE(ref.basis.size() == n_aed);
  std::vector<bool> active(ref.cols(), true);
  std::size_t n_zero = 0;
  for (std::size_t j = 1; j < ref.basis.size(); ++j) {
    bool zero = true;
    for (ld v : ref.basis[j]) zero = zero && v == 0;
    if (zero) {
      active[j] = false;
      ++n_zero;
    }
  }
  REQUIRE(n_zero >= 1);
  Rows A;
  std::vector<ld> T;
  build_rows(ref, t.tr->dc, t.db, A, T);
  const LsSolution sol = least_squares(A, T, active);
  INFO("reference rcond " << static_cast<double>(sol.rcond));
  REQUIRE(sol.rcond > 1e-9);
  const t_type w = t.tr->model->get_weights();
  REQUIRE(w.size() == n_aed);
  ld wmax = 0;
  for (std::size_t j = 0; j < w.size(); ++j) wmax = std::max(wmax, std::fabs(ld(w(j))));
  for (std::size_t j = 0; j < w.size(); ++j)
    if (!active[j]) {
      INFO("zero basis vector " << j);
      CHECK(std::fabs(ld(w(j))) < 1e-12L * wmax);
    }
  INFO("weights   " << to_text(w) << "\nreference " << to_text(sol.w));
  CHECK(rel_diff(w, sol.w) < 1e-8);
}

/* Pins: with BIAS true (ATOM Ta: the descriptor's column 0 is the bias, 1
 * for every atom; DSIZE 4), LAMBDA 0.01 and OALGO 1, each function's fit
 * equals the reference ridge solution that leaves the function's intercept
 * columns out of the penalty, to 1e-8 relative (measured: 4e-15): column 0 for
 * Kern_Polynomial, Kern_Quadratic, Kern_LQ and Kern_Sigmoid (functions of
 * b . x whose basis vector 0 is e0: K(e0, x) is the same for every atom)
 * and for BF_Polynomial2 (bias x bias); none for Kern_RBF; and a check
 * that the reference with the other choice (column 0 penalised, or for
 * Kern_RBF left out) differs by more than 1e-4 (measured: 1.1e-3 ..
 * 0.27), so the comparison tells the two apart.
 * Before the fix: training threw (the ColumnLayout check). With that check
 * alone fixed (seen in a scratch build), M_KRR left rotated column 0
 * (EKM) out of the penalty instead, whatever the kernel: the weights
 * differed from the reference by 0.3 % (Kern_Polynomial) to 14 %
 * (Kern_RBF); Kern_Sigmoid lost e0 from its basis (K(e0, e0) < 0, SBASIS
 * 4) and the process aborted. */
TEST_CASE("The intercept columns of each non-linear function are left out "
          "of the ridge penalty",
          "[nonlinear][train][intercept]") {
  PeriodicTable::initialize();
  const ld lambda = 0.01L;
  for (const ModelSpec &m : nonlinear_models()) {
    DYNAMIC_SECTION(m.name) {
      Context c = make_context(m, /*bias=*/true);
      set_key(c, "LAMBDA", "0.01");
      Trained t(c);
      REQUIRE_NOTHROW(t.train());
      const Reference ref = reference_of(*t.tr, m);
      REQUIRE(ref.dim == 4);
      if (m.kernel) {  // basis vector 0 is e0
        CHECK(ref.basis[0][0] == 1);
        for (std::size_t d = 1; d < ref.dim; ++d) CHECK(ref.basis[0][d] == 0);
      }
      const std::size_t n_int = m.model[1] == "Kern_RBF" ? 0 : 1;
      Rows A;
      std::vector<ld> T;
      build_rows(ref, t.tr->dc, t.db, A, T);
      const LsSolution sol = ridge(A, T, lambda, n_int);
      const LsSolution other = ridge(A, T, lambda, 1 - n_int);
      const t_type w = t.tr->model->get_weights();
      INFO("intercept columns " << n_int);
      INFO("weights   " << to_text(w) << "\nreference " << to_text(sol.w));
      CHECK(rel_diff(w, sol.w) < 1e-8);
      t_type wo(other.w.size());
      for (std::size_t j = 0; j < other.w.size(); ++j)
        wo(j) = static_cast<double>(other.w[j]);
      CHECK(rel_diff(wo, sol.w) > 1e-4);
    }
  }
}

/* Pins: OALGO 3 (SVD, uniform ridge) with BIAS true, ATOM Ta and LAMBDA
 * 0.01 trains M_KRR Kern_RBF, which has no intercept column, to the ridge
 * solution penalising every column (1e-8 relative); it still refuses M_KRR
 * Kern_Quadratic, whose column 0 is an intercept, with the message saying
 * so ("does not support intercept exclusion", "1 intercept column").
 * Before the fix: both threw the ColumnLayout check, so the Kern_RBF fit
 * and the Kern_Quadratic message CHECKs failed. (Without it, OALGO 3
 * refused every model with BIAS true, ATOM and LAMBDA != 0.) */
TEST_CASE("OALGO 3 with BIAS true trains a kernel without intercept columns "
          "and refuses one with",
          "[nonlinear][train][intercept]") {
  PeriodicTable::initialize();
  {
    const ModelSpec &m = model_named("M_KRR Kern_RBF");
    Context c = make_context(m, /*bias=*/true);
    set_key(c, "LAMBDA", "0.01");
    set_key(c, "OALGO", "3");
    Trained t(c);
    REQUIRE_NOTHROW(t.train());
    const Reference ref = reference_of(*t.tr, m);
    Rows A;
    std::vector<ld> T;
    build_rows(ref, t.tr->dc, t.db, A, T);
    const LsSolution sol = ridge(A, T, 0.01L, 0);
    const t_type w = t.tr->model->get_weights();
    INFO("weights   " << to_text(w) << "\nreference " << to_text(sol.w));
    CHECK(rel_diff(w, sol.w) < 1e-8);
  }
  {
    const ModelSpec &m = model_named("M_KRR Kern_Quadratic");
    Context c = make_context(m, /*bias=*/true);
    set_key(c, "LAMBDA", "0.01");
    set_key(c, "OALGO", "3");
    Trained t(c);
    std::string msg;
    try {
      t.train();
    } catch (const std::exception &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(msg.find("does not support intercept exclusion") != std::string::npos);
    CHECK(msg.find("1 intercept column") != std::string::npos);
  }
}

namespace {

/* The documented draw, written out here: the first k steps of a
 * Fisher-Yates shuffle of 0 .. n-1 driven by the raw output of
 * std::mt19937_64 with its default seed, each bounded draw by rejection
 * below 2^64 mod (n - i). */
std::vector<std::size_t> documented_draw(std::size_t n, std::size_t k) {
  std::mt19937_64 g;  // default seed 5489
  std::vector<std::size_t> idx(n);
  for (std::size_t i = 0; i < n; ++i) idx[i] = i;
  for (std::size_t i = 0; i < k; ++i) {
    const std::uint64_t range = n - i;
    const std::uint64_t reject_below = (0 - range) % range;
    std::uint64_t r;
    do {
      r = g();
    } while (r < reject_below);
    std::swap(idx[i], idx[i + static_cast<std::size_t>(r % range)]);
  }
  idx.resize(k);
  return idx;
}

}  // namespace

/* Pins: the basis a kernel draws is the same in two Trainers on the same
 * data (BASIS bit for bit, written with 17 digits), and it is the
 * documented draw: column 0 is e0 = (1, 0, 0), and columns 1 .. 4 are the
 * AEDs at positions 1, 9, 12, 19 of the list of the 21 training AEDs
 * (structure by structure, atom by atom), the draw of std::mt19937_64 with
 * its default seed 5489 (whose 10000th output the C++ standard fixes:
 * 9981545732273789042, checked) through a partial Fisher-Yates shuffle with
 * rejection-bounded draws (written out in this test; the positions were
 * also obtained with an independent Python implementation of the
 * MT19937-64 reference algorithm).
 * Before the fix: training threw (the ColumnLayout check). With that check
 * alone fixed (seen in a scratch build), every training drew its basis
 * with std::random_device and std::shuffle (implementation-defined), and
 * the two Trainers wrote different BASIS lines. */
TEST_CASE("The basis draw is deterministic and portable",
          "[nonlinear][train][basis]") {
  PeriodicTable::initialize();
  {
    std::mt19937_64 g;
    g.discard(9999);
    REQUIRE(g() == 9981545732273789042ULL);
  }
  const std::vector<std::size_t> drawn = documented_draw(21, kSbasis - 1);
  CHECK(drawn == std::vector<std::size_t>{1, 9, 12, 19});

  const ModelSpec &m = model_named("M_KRR Kern_RBF");
  Trained t1(make_context(m)), t2(make_context(m));
  REQUIRE_NOTHROW(t1.train());
  REQUIRE_NOTHROW(t2.train());
  const Context p1 = t1.tr->model->get_param_file();
  const Context p2 = t2.tr->model->get_param_file();
  CHECK(p1("BASIS") == p2("BASIS"));

  const Reference ref = reference_of(*t1.tr, m);
  REQUIRE(ref.basis.size() == kSbasis);
  CHECK(ref.basis[0] == std::vector<ld>{1, 0, 0});
  std::vector<std::vector<double>> aeds;
  for (std::size_t s = 0; s < t1.db.size(); ++s) {
    const StDescriptors sd = t1.tr->dc.calc(t1.db(s));
    for (std::size_t a = 0; a < sd.naed(); ++a) {
      std::vector<double> v;
      for (std::size_t d = 0; d < sd.dim(); ++d) v.push_back(sd.get_aed(a)[d]);
      aeds.push_back(v);
    }
  }
  REQUIRE(aeds.size() == 21);
  for (std::size_t j = 1; j < kSbasis; ++j) {
    INFO("basis column " << j << " = AED " << drawn[j - 1]);
    for (std::size_t d = 0; d < 3; ++d)
      CHECK(static_cast<double>(ref.basis[j][d]) == aeds[drawn[j - 1]][d]);
  }
}

/* Pins: SBASIS 22 (e0 and all 21 training AEDs) trains M_KRR Kern_RBF;
 * SBASIS 1, 23 (the 21 training AEDs allow at most 22) and 0 are refused
 * with a message naming SBASIS and the limit ("at least 2", "at most
 * 22").
 * Before the fix: SBASIS 22 threw "The number of requested basis vectors
 * is larger than the amount of available AEDs" (the check wanted SBASIS
 * <= 21 although the draw used only SBASIS - 1 AEDs); SBASIS 1 built the
 * basis e0 alone and the training threw the ColumnLayout check; SBASIS 23
 * threw the message above (no SBASIS, no limit); SBASIS 0 wrote b(0,0)
 * into a basis of 0 columns (SIGSEGV, which ends the run; it is the last
 * section). */
TEST_CASE("SBASIS below 2 or above the number of AEDs + 1 is refused, "
          "naming SBASIS",
          "[nonlinear][train][basis]") {
  PeriodicTable::initialize();
  const ModelSpec &m = model_named("M_KRR Kern_RBF");
  SECTION("SBASIS 22") {
    Context c = make_context(m);
    set_key(c, "SBASIS", "22");
    Trained t(c);
    REQUIRE_NOTHROW(t.train());
    CHECK(t.tr->model->get_weights().size() == 22);
  }
  for (const char *s : {"1", "23", "0"}) {
    DYNAMIC_SECTION("SBASIS " << s) {
      Context c = make_context(m);
      set_key(c, "SBASIS", s);
      Trained t(c);
      std::string msg;
      try {
        t.train();
      } catch (const std::exception &e) {
        msg = e.what();
      }
      INFO("message: " << msg);
      CHECK(msg.find(std::string("SBASIS ") + s) != std::string::npos);
      CHECK(msg.find(std::string(s) == "23" ? "at most 22" : "at least 2") !=
            std::string::npos);
    }
  }
}

namespace {

/* MODEL M_KRR Kern_RBF 2: implementation 2, energies only. */
const ModelSpec kRbf2 = {"M_KRR Kern_RBF 2", {"M_KRR", "Kern_RBF", "2"},
                         {"1.0"}, true};

/* The mean AED of each structure of the data (implementation 2's
 * candidates), with the descriptors of the Trainer. */
std::vector<std::vector<ld>> mean_aeds(Trained &t) {
  std::vector<std::vector<ld>> out;
  for (std::size_t s = 0; s < t.db.size(); ++s) {
    const StDescriptors sd = t.tr->dc.calc(t.db(s));
    std::vector<ld> m(sd.dim(), 0);
    for (std::size_t a = 0; a < sd.naed(); ++a)
      for (std::size_t d = 0; d < sd.dim(); ++d)
        m[d] += ld(sd.get_aed(a)[d]) / ld(sd.naed());
    out.push_back(m);
  }
  return out;
}

}  // namespace

/* Pins: implementation 2 (MODEL M_KRR Kern_RBF 2, energies only) with
 * SBASIS 5 on the six structures: two Trainers on the same data write the
 * same BASIS and WEIGHTS (17 digits, bit for bit); its basis is the
 * documented draw: column 0 is e0 = (1, 0, 0) and columns 1 .. 4 are the
 * mean AEDs of structures 4, 0, 2, 1 (draw_candidates(6, 4): the draw of
 * the basis test above on 6 candidates, also obtained with an independent
 * Python implementation of MT19937-64), each component to 1e-14
 * (relative, or absolute below 1); and the weights solve
 * (K + LAMBDA I) w = t with K_ij = exp(-|b_i - b_j|^2 / 2) (Kern_RBF, sigma
 * 1, LAMBDA 0), t_0 = 0 and t_j the energy per atom of the structure of
 * column j, to 1e-8 relative (the reference: Householder QR in long
 * double; K full rank, rcond > 1e-9).
 * Before the fix (the old prep_basis_for_krr restored in a scratch
 * build): the structures were drawn with std::random_device and
 * std::shuffle (implementation-defined), so the two Trainers wrote
 * different BASIS and WEIGHTS lines (unless the draws coincided, 1 in
 * 360) and the columns were not structures 4, 0, 2, 1, so the weights,
 * fitted to the structures drawn, missed the reference built on the
 * documented ones (5e-2 relative in that run). */
TEST_CASE("Implementation 2 draws its basis by the documented draw and "
          "solves (K + lambda I) w = t",
          "[nonlinear][train][basis]") {
  PeriodicTable::initialize();
  const std::vector<std::size_t> drawn = documented_draw(6, kSbasis - 1);
  REQUIRE(drawn == std::vector<std::size_t>{4, 0, 2, 1});

  Trained t1(make_context(kRbf2)), t2(make_context(kRbf2));
  REQUIRE_NOTHROW(t1.train());
  REQUIRE_NOTHROW(t2.train());
  const Context p1 = t1.tr->model->get_param_file();
  const Context p2 = t2.tr->model->get_param_file();
  CHECK(p1("MODEL") == std::vector<std::string>{"M_KRR", "Kern_RBF", "2"});
  CHECK(p1("BASIS") == p2("BASIS"));
  CHECK(p1("WEIGHTS") == p2("WEIGHTS"));

  const Reference ref = reference_of(*t1.tr, kRbf2);
  REQUIRE(ref.basis.size() == kSbasis);
  CHECK(ref.basis[0] == std::vector<ld>{1, 0, 0});
  const std::vector<std::vector<ld>> means = mean_aeds(t1);
  REQUIRE(means.size() == 6);
  for (std::size_t j = 1; j < kSbasis; ++j)
    for (std::size_t d = 0; d < 3; ++d) {
      INFO("basis column " << j << " = structure " << drawn[j - 1]
                           << ", component " << d);
      CHECK(static_cast<double>(ref.basis[j][d]) ==
            Approx(static_cast<double>(means[drawn[j - 1]][d]))
                .epsilon(1e-14)
                .margin(1e-14));
    }

  // (K + LAMBDA I) w = t on the read-back basis, LAMBDA 0.
  Rows K(kSbasis, std::vector<ld>(kSbasis));
  std::vector<ld> rhs(kSbasis, 0);
  std::vector<ld> phi;
  std::vector<std::vector<ld>> dphi;
  for (std::size_t i = 0; i < kSbasis; ++i) {
    tadah::core::aed_type bi(3);
    for (std::size_t d = 0; d < 3; ++d)
      bi[d] = static_cast<double>(ref.basis[i][d]);
    ref.eval(bi, phi, dphi);  // phi[j] = K(b_j, b_i)
    for (std::size_t j = 0; j < kSbasis; ++j) K[i][j] = phi[j];
  }
  for (std::size_t j = 1; j < kSbasis; ++j) {
    const Structure &st = t1.db(drawn[j - 1]);
    rhs[j] = ld(st.energy) / ld(st.natoms());
  }
  const LsSolution sol = least_squares(K, rhs);
  INFO("reference rcond (min/max |R_kk|) " << static_cast<double>(sol.rcond));
  REQUIRE(sol.rcond > 1e-9);
  const t_type w = t1.tr->model->get_weights();
  INFO("weights   " << to_text(w));
  INFO("reference " << to_text(sol.w));
  CHECK(rel_diff(w, sol.w) < 1e-8);
}

/* Pins: implementation 2 (MODEL M_KRR Kern_RBF 2) with SBASIS 7 (e0 and
 * all 6 structures) trains, with 7 weights; SBASIS 1, 8 (the 6 structures
 * allow at most 7) and 0 are refused with a message naming SBASIS and the
 * limit ("at least 2", "at most 7").
 * Before the fix (the old prep_basis_for_krr restored in a scratch
 * build): SBASIS 7 threw "The number of requested basis vectors is larger
 * than the amount of available AEDs" (the check wanted SBASIS <= 6,
 * although the draw used only SBASIS - 1 structures); SBASIS 1 trained on
 * e0 alone (no throw); SBASIS 8 threw that message, which names neither
 * SBASIS nor the limit; SBASIS 0 wrote b(0,0) into a basis of 0 columns
 * (SIGSEGV, which ends the run; it is the last section). */
TEST_CASE("Implementation 2: SBASIS below 2 or above the number of "
          "structures + 1 is refused, naming SBASIS",
          "[nonlinear][train][basis]") {
  PeriodicTable::initialize();
  SECTION("SBASIS 7") {
    Context c = make_context(kRbf2);
    set_key(c, "SBASIS", "7");
    Trained t(c);
    REQUIRE_NOTHROW(t.train());
    CHECK(t.tr->model->get_weights().size() == 7);
  }
  for (const char *s : {"1", "8", "0"}) {
    DYNAMIC_SECTION("SBASIS " << s) {
      Context c = make_context(kRbf2);
      set_key(c, "SBASIS", s);
      Trained t(c);
      std::string msg;
      try {
        t.train();
      } catch (const std::exception &e) {
        msg = e.what();
      }
      INFO("message: " << msg);
      CHECK(msg.find(std::string("SBASIS ") + s) != std::string::npos);
      CHECK(msg.find(std::string(s) == "8" ? "at most 7" : "at least 2") !=
            std::string::npos);
    }
  }
}

/* Pins: M_KRR with a non-linear kernel refuses train(StDescriptorsDB,
 * StructureDB), the training from precomputed descriptors, which draws no
 * basis: the message names the kernel and says no basis vectors are drawn.
 * Kern_Linear trains on that path (a pin).
 * Before the fix: Kern_RBF "trained" on a design matrix of 0 columns (no
 * basis vector): the EKM projection called DGEMM with an empty map (MKL:
 * "Parameter 10 was incorrect on entry to DGEMM"), and no exception was
 * thrown, so the message CHECKs failed. */
TEST_CASE("M_KRR refuses to train a non-linear kernel from precomputed "
          "descriptors",
          "[nonlinear][train]") {
  PeriodicTable::initialize();
  {
    const ModelSpec &m = model_named("M_KRR Kern_RBF");
    Trained t(make_context(m));
    t.tr->nnf.calc(t.db, t.tr->context);
    tadah::mlip::StDescriptorsDB sdb = t.tr->dc.calc(t.db);
    std::string msg;
    try {
      t.tr->model->train(sdb, t.db);
    } catch (const std::exception &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(msg.find("Kern_RBF") != std::string::npos);
    CHECK(msg.find("draws no basis") != std::string::npos);
  }
  {
    const ModelSpec lin = {"M_KRR Kern_Linear", {"M_KRR", "Kern_Linear", "1"}, {}, false};
    Trained t(make_context(lin));
    t.tr->nnf.calc(t.db, t.tr->context);
    tadah::mlip::StDescriptorsDB sdb = t.tr->dc.calc(t.db);
    REQUIRE_NOTHROW(t.tr->model->train(sdb, t.db));
    CHECK(t.tr->model->get_weights().size() == 3);
  }
}

namespace {

/* An M_KRR Kern_RBF potential written here (not trained): the training
 * configuration with SBASIS 5, `n_basis` BASIS values (0.1 0.2 ...) and
 * `n_weights` WEIGHTS; DSIZE is left to the DescriptorsCalc, as when a
 * potential file is read. */
Context handmade_rbf_potential(std::size_t n_basis, std::size_t n_weights) {
  Context c = make_context(model_named("M_KRR Kern_RBF"));
  for (std::size_t v = 0; v < n_basis; ++v) c.add("BASIS", 0.1 * (v + 1));
  for (std::size_t v = 0; v < n_weights; ++v) c.add("WEIGHTS", 0.5);
  return c;
}

/* A potential loaded as `tadah predict` loads it: DC_Selector, the
 * DescriptorsCalc on its descriptors (which sets DSIZE), the function and
 * the model, built from the potential's context (a Trainer builds a model
 * to be fitted, from the configuration without WEIGHTS and BASIS). */
struct LoadedPotential {
  Context ctx;
  tadah::models::DC_Selector dcs;
  DescriptorsCalc<> dc;
  std::unique_ptr<tadah::mlip::DM_Function_Base> fb;
  std::unique_ptr<tadah::mlip::M_Tadah_Base> model;
  explicit LoadedPotential(const Context &pot)
      : ctx(pot), dcs(ctx), dc(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb),
        fb(tadah::core::factory<tadah::mlip::DM_Function_Base,
                                const Context &>(
            ctx.get<std::string>("MODEL", 1), ctx)),
        model(tadah::core::factory<tadah::mlip::M_Tadah_Base,
                                   tadah::mlip::DM_Function_Base &, Context &>(
            ctx.get<std::string>("MODEL", 0), *fb, ctx)) {}
};

/* The message of the exception loading the potential `c` as `tadah
 * predict` does (LoadedPotential) throws, "" when none. */
std::string load_error(Context c) {
  try {
    LoadedPotential loaded(c);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

}  // namespace

/* Pins: a trained M_KRR Kern_RBF potential (get_param_file, DSIZE removed
 * as a file would not have it) loads as `tadah predict` loads it
 * (LoadedPotential) and predicts what the trained model predicts (1e-14
 * relative), and the loaded model writes the potential back with SBASIS 5
 * and the trained BASIS, value for value; a potential with 5 x 3 BASIS
 * values and 5 WEIGHTS loads (a pin of the hand-made potential); with BASIS
 * holding SBASIS (5) vectors of 4 values instead of 3 (the length is a
 * multiple of SBASIS, so the kernel reads it) it loads, and predicting
 * with it is refused with a message naming BASIS, SBASIS and DSIZE ("of 4
 * values", "has 3 components (DSIZE)"): the length is checked where the
 * descriptors meet the basis; with 4 WEIGHTS for 5 basis vectors it is
 * refused when loaded, naming WEIGHTS and SBASIS ("4 WEIGHTS", "SBASIS
 * 5").
 * Before the fix: the training threw (the ColumnLayout check), so the
 * first section failed; the potential with 4-value vectors and the one
 * with 4 WEIGHTS loaded without an error (the first then read past every
 * 3-value descriptor in K(b, x), the second past its weights). After that,
 * with M_KRR's get_param_file writing M_KRR's own copy of the basis (seen
 * in a scratch build), the loaded model wrote SBASIS 0 and no BASIS (that
 * copy is empty in a model loaded from a potential); and with the length
 * checked when M_KRR was built (the first version of this fix), building
 * the Trainer on the 4-value potential threw, so its REQUIRE_NOTHROW
 * failed. Until the Trainer built its model from the configuration without
 * WEIGHTS and BASIS (a model to be fitted), the potential was loaded into a
 * Trainer; the checks are the same, and with LoadedPotential they pass
 * before that change as well (a pin: tadah predict has always loaded a
 * potential so). */
TEST_CASE("A non-linear KRR potential's BASIS must match its descriptor and "
          "WEIGHTS",
          "[nonlinear][load]") {
  PeriodicTable::initialize();
  SECTION("the potential as trained") {
    const ModelSpec &m = model_named("M_KRR Kern_RBF");
    Trained t(make_context(m));
    REQUIRE_NOTHROW(t.train());
    Context c = t.tr->model->get_param_file();
    c.remove("DSIZE");  // as read from a file: DescriptorsCalc sets it
    std::unique_ptr<LoadedPotential> loaded;
    REQUIRE_NOTHROW(loaded = std::make_unique<LoadedPotential>(c));
    Context pc = t.tr->context;
    const StructureDB a = t.tr->model->predict(pc, t.db, t.tr->dc);
    Context lc = loaded->ctx;
    const StructureDB b = loaded->model->predict(lc, t.db, loaded->dc);
    for (std::size_t s = 0; s < t.db.size(); ++s)
      CHECK(b(s).energy == Approx(a(s).energy).epsilon(1e-14));
    // The loaded model writes the basis it predicts with.
    const Context written = loaded->model->get_param_file();
    REQUIRE(written.contains("SBASIS"));
    CHECK(written.get<std::size_t>("SBASIS") == kSbasis);
    REQUIRE(written.contains("BASIS"));
    REQUIRE(written.size("BASIS") == c.size("BASIS"));
    REQUIRE(c.size("BASIS") == kSbasis * 3);
    for (std::size_t v = 0; v < c.size("BASIS"); ++v) {
      INFO("BASIS value " << v);
      CHECK(written.get<double>("BASIS", v) == c.get<double>("BASIS", v));
    }
  }
  SECTION("5 vectors of 3 values and 5 WEIGHTS (a pin)") {
    CHECK(load_error(handmade_rbf_potential(kSbasis * 3, kSbasis)) == "");
  }
  SECTION("BASIS vectors one value too long") {
    Context c = handmade_rbf_potential(kSbasis * 4, kSbasis);
    std::unique_ptr<LoadedPotential> loaded;
    REQUIRE_NOTHROW(loaded = std::make_unique<LoadedPotential>(c));
    StructureDB db = training_db();
    NNFinder().calc(db, loaded->ctx);
    std::string msg;
    try {
      Context lc = loaded->ctx;
      (void)loaded->model->predict(lc, db, loaded->dc);
    } catch (const std::exception &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(msg.find("BASIS") != std::string::npos);
    CHECK(msg.find("of 4 values") != std::string::npos);
    CHECK(msg.find("has 3 components (DSIZE)") != std::string::npos);
  }
  SECTION("WEIGHTS fewer than the basis vectors") {
    const std::string msg =
        load_error(handmade_rbf_potential(kSbasis * 3, 4));
    INFO("message: " << msg);
    CHECK(msg.find("4 WEIGHTS") != std::string::npos);
    CHECK(msg.find("SBASIS 5") != std::string::npos);
  }
}

/* Pins: an M_KRR<DM_Kern_RBF> built from a trained Kern_RBF potential
 * before the DescriptorsCalc<D2_Blip, DM_Dummy, Cut_Cos, Cut_Cos> of that
 * potential, the order of the API example (Tadah.MLIP examples/API/ex_2),
 * builds (the context has no DSIZE yet) and predicts, through
 * predict(context, StructureDB, DescriptorsCalc), the energies, forces and
 * stresses the trained model predicts (1e-14 relative, 1e-14 absolute);
 * built so from a potential whose BASIS holds SBASIS (5) vectors of 4
 * values, it builds too, and predicting is refused with the message
 * naming BASIS and DSIZE ("of 4 values", "has 3 components (DSIZE)").
 * Before the fix: M_KRR's constructor checked the length of the basis
 * vectors against DSIZE and threw std::logic_error ("the descriptor size
 * (DSIZE) is not known ... build the DescriptorsCalc on this context
 * before the model") in both sections, so both REQUIRE_NOTHROWs failed (on
 * v1-dev, without that check, the trained potential predicted). */
TEST_CASE("A non-linear KRR potential built before its DescriptorsCalc "
          "predicts, and a BASIS of another length is refused when it "
          "predicts",
          "[nonlinear][load]") {
  PeriodicTable::initialize();
  using Model = tadah::mlip::M_KRR<tadah::mlip::DM_Kern_RBF>;
  using DC = DescriptorsCalc<tadah::models::D2_Blip, tadah::models::DM_Dummy,
                             tadah::models::Cut_Cos, tadah::models::Cut_Cos>;
  SECTION("the potential as trained") {
    const ModelSpec &m = model_named("M_KRR Kern_RBF");
    Trained t(make_context(m));
    REQUIRE_NOTHROW(t.train());
    Context pc = t.tr->context;
    const StructureDB a = t.tr->model->predict(pc, t.db, t.tr->dc);
    Context c = t.tr->model->get_param_file();
    c.remove("DSIZE");  // as read from a file: DescriptorsCalc sets it
    std::unique_ptr<Model> model;
    REQUIRE_NOTHROW(model = std::make_unique<Model>(c));
    StructureDB db = training_db();
    NNFinder().calc(db, c);
    DC dc(c);
    const StructureDB b = model->predict(c, db, dc);
    REQUIRE(b.size() == a.size());
    for (std::size_t s = 0; s < a.size(); ++s) {
      INFO("structure " << s);
      CHECK(b(s).energy == Approx(a(s).energy).epsilon(1e-14));
      for (std::size_t i = 0; i < db(s).natoms(); ++i)
        for (int k = 0; k < 3; ++k)
          CHECK(b(s).atoms[i].force[k] ==
                Approx(a(s).atoms[i].force[k]).epsilon(1e-14).margin(1e-14));
      for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
          CHECK(b(s).stress(x, y) ==
                Approx(a(s).stress(x, y)).epsilon(1e-14).margin(1e-14));
    }
  }
  SECTION("BASIS vectors one value too long") {
    Context c = handmade_rbf_potential(kSbasis * 4, kSbasis);
    std::unique_ptr<Model> model;
    REQUIRE_NOTHROW(model = std::make_unique<Model>(c));
    StructureDB db = training_db();
    NNFinder().calc(db, c);
    DC dc(c);
    std::string msg;
    try {
      (void)model->predict(c, db, dc);
    } catch (const std::exception &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(msg.find("BASIS") != std::string::npos);
    CHECK(msg.find("of 4 values") != std::string::npos);
    CHECK(msg.find("has 3 components (DSIZE)") != std::string::npos);
  }
}

/* Pins: for M_KRR with a non-linear kernel the weights belong to basis
 * vectors drawn from the training data, so FIXINDEX/FIXWEIGHT alone do not
 * make a model: Trainer::provision_fixed_weights refuses, saying the basis
 * is drawn from the training data, and classify_weight_source refuses a
 * configuration without a training DBFILE, naming DBFILE; with a DBFILE a
 * FIXINDEX naming weight 6 of SBASIS 5 is out of range ("FIXINDEX: index 6
 * is outside 1-5 (the model has 5 weights)", core::parse_fixindex's
 * message, the one every site gives).
 * Before the fix: provision_fixed_weights threw "FIXINDEX entry 1 is out
 * of range: this model has 0 weights" (the kernel has no basis before
 * training, its width was 0); classify_weight_source with K = 5 and every
 * weight pinned returned FixedOnlyNoData without a DBFILE (a model with
 * weights and no basis). The range check was right (a pin), with MLIP's
 * own text, "FIXINDEX entry 6 is out of range: this model has 5 weights
 * (valid range 1..5).", which this case checked for "this model has 5
 * weights" until fix/fixindex-strict. */
TEST_CASE("Non-linear KRR weights cannot be provisioned without training "
          "data",
          "[nonlinear][fixweight]") {
  PeriodicTable::initialize();
  const ModelSpec &m = model_named("M_KRR Kern_RBF");
  Context c = make_context(m);
  c.add("FIXINDEX", "1-5");
  for (int k = 0; k < 5; ++k) c.add("FIXWEIGHT", 0.1 * (k + 1));
  {
    MLIPWorkspaceManager ws;
    Context cc = c;
    Trainer tr(cc, ws);
    std::string msg;
    try {
      tr.provision_fixed_weights();
    } catch (const std::exception &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(msg.find("drawn from the training data") != std::string::npos);
  }
  {
    std::string msg;
    try {
      (void)tadah::mlip::classify_weight_source(c, kSbasis);
    } catch (const std::exception &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(msg.find("DBFILE") != std::string::npos);
    CHECK(msg.find("Kern_RBF") != std::string::npos);
  }
  {
    Context cd = make_context(m);
    cd.add("DBFILE", "train.db");
    cd.add("FIXINDEX", "6");
    cd.add("FIXWEIGHT", 0.5);
    CHECK_THROWS_WITH(
        tadah::mlip::classify_weight_source(cd, kSbasis),
        "FIXINDEX: index 6 is outside 1-5 (the model has 5 weights)");
  }
}

