// test_fit_balance.cpp
//
// The balance table of a fit (fit_balance.h): per quantity and per dataset
// the observations n of the fit, the RMSE in the unit of the quantity's
// metric over every entry of priority > 0 (meV/atom, eV/Å, GPa over the
// nine entries of a stress tensor: those of tadah predict -A), the root
// mean square of residual / error scale over the same entries, the share
// of the sum the fit minimises and the effective parameters gamma, with the
// error scales and the prior's row (the LAMBDA penalty and the rest of the
// rank); its lines, at most 78 characters wide.
//
// The references are independent of the code under test: a solve by hand
// of a fit whose only coefficients the data determine are the intercepts of
// two species (2 x 2 algebra in long double), an SVD of the weighted design
// matrix (LAPACK dgesvd) for the sum of the effective parameters, sum_kept
// s_i^2 / (s_i^2 + lambda), the predictions of the trained model for the
// RMSE of each quantity in its metric's unit (Analytics, the metrics of
// tadah predict -A), and the lines of a table written out by hand.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/lapack.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/design_matrix.h>
#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_linear.h>
#include <tadah/mlip/fit_balance.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/training_record.h>
#include <tadah/models/dc_selector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using Catch::Matchers::Contains;
using Catch::Matchers::StartsWith;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::FitBalance;
using tadah::mlip::FitBalanceRow;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

using LD = long double;

void set_key(Context &c, const std::string &key, const std::string &value) {
  if (c.has_stored(key)) c.remove(key);
  c.add(key, value);
}

/* The row of quantity @p q (0 energy, 1 force, 2 stress) and dataset @p d
 * (0: every dataset) of @p b; fails the test when there is none. */
const FitBalanceRow &row_of(const FitBalance &b, std::size_t q, std::size_t d) {
  const FitBalanceRow *r = b.row(q, d);
  INFO("quantity " << q << ", dataset " << d);
  REQUIRE(r != nullptr);
  return *r;
}

/* The characters of @p s as a terminal shows them (UTF-8 code points). */
std::size_t display_width(const std::string &s) {
  std::size_t n = 0;
  for (unsigned char ch : s)
    if ((ch & 0xC0) != 0x80) ++n;
  return n;
}

/* The widest line of a table: 78 characters, 80 in a training record after
 * its "# " (a terminal; the documentation's code blocks). */
constexpr std::size_t kWidth = 78;

/* Fails the test for a line of @p lines wider than kWidth. */
void check_width(const std::vector<std::string> &lines) {
  for (const std::string &l : lines) {
    INFO("'" << l << "'");
    CHECK(display_width(l) <= kWidth);
  }
}

/* The text of the paragraph that begins at lines[@p first]: that line and
 * the lines after it that continue it (four blanks), joined by a blank. */
std::string paragraph(const std::vector<std::string> &lines,
                      std::size_t first) {
  REQUIRE(first < lines.size());
  std::string text = lines[first];
  for (std::size_t k = first + 1;
       k < lines.size() && lines[k].rfind("    ", 0) == 0; ++k)
    text += " " + lines[k].substr(lines[k].find_first_not_of(' '));
  return text;
}

// ---------------------------------------------------------------------------
// The two-species fit of the independent oracle
// ---------------------------------------------------------------------------

/* A cubic cell of edge 30 Å with the atoms @p species 10 Å apart (no pair
 * within the cutoff of 3 Å: every descriptor column is zero, and the energy
 * row is the number of atoms of each species, the intercept columns). */
Structure isolated_atoms(const std::vector<std::string> &species,
                         double energy, double eweight) {
  Structure st;
  const double a = 30.0;
  st.cell.load(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  const double pos[4][3] = {
      {0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}};
  for (std::size_t i = 0; i < species.size(); ++i)
    st.add_atom(Atom(Element(species[i]), pos[i][0], pos[i][1], pos[i][2],
                     0.0, 0.0, 0.0));
  st.energy = energy;
  st.eweight = eweight;
  return st;
}

Context intercept_context(const std::string &lambda, const std::string &enorm) {
  Context c;
  c.add("ATOM", "Nb Ta");
  c.add("WATOM", "1 1");
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 2 2 * *");
  c.add("CGRID2B", "1.0 2.0");
  c.add("SGRID2B", "1.0 1.0");
  c.add("TYPEMB", "DM_Dummy * *");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 3.0);
  c.add("RCUTMB", 3.0);
  c.add("DIMER", "false 0 false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("BIAS", "true");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("OALGO", 1);
  c.add("LAMBDA", lambda);
  c.add("ESIGMA", 2);
  c.add("ENORM", enorm);
  c.add("WDBFILE", "1 3");
  return c;
}

/* Two datasets: {Ta2, NbTa} and {Nb3, NbTa3}, eweight 1, 2, 0.5, 1; with
 * WDBFILE 1 3 the priorities are 1, 2, 1.5 and 3. */
StructureDB intercept_db() {
  StructureDB db;
  db.add(isolated_atoms({"Ta", "Ta"}, -16.2, 1.0));
  db.add(isolated_atoms({"Nb", "Ta"}, -17.9, 2.0));
  db.add(isolated_atoms({"Nb", "Nb", "Nb"}, -30.1, 0.5));
  db.add(isolated_atoms({"Nb", "Ta", "Ta", "Ta"}, -41.0, 1.0));
  db.dbidx = {0, 2, 4};
  return db;
}

/* The fit of the intercepts by hand: W_s = p_s / sigma_s^2, sigma_s =
 * N_s^q ESIGMA/1000 eV, phi_s = (atoms of Nb, atoms of Ta);
 * H = sum_s W_s phi_s phi_s^T, b = H^-1 sum_s W_s phi_s E_s, r_s = E_s -
 * phi_s . b; per dataset gamma_d = tr(H^-1 H_d), the RMSE of the energy
 * per atom in meV/atom, the RMS of r_s / sigma_s and the share of
 * sum_s W_s r_s^2. */
struct Oracle {
  std::array<LD, 3> gamma{}, rmse{}, rms_z{}, share{};  // [0] all, [1], [2]
};

Oracle intercept_oracle(bool sqrt_norm) {
  const LD esigma = 2.0L / 1000.0L;
  const int nb[4] = {0, 1, 3, 1}, ta[4] = {2, 1, 0, 3};
  const LD energy[4] = {-16.2L, -17.9L, -30.1L, -41.0L};
  const LD p[4] = {1.0L, 2.0L, 1.5L, 3.0L};
  const int ds[4] = {1, 1, 2, 2};
  LD W[4], sigma[4], N[4];
  for (int s = 0; s < 4; ++s) {
    N[s] = static_cast<LD>(nb[s] + ta[s]);
    sigma[s] = (sqrt_norm ? std::sqrt(N[s]) : N[s]) * esigma;
    W[s] = p[s] / (sigma[s] * sigma[s]);
  }
  LD H[2][2] = {{0, 0}, {0, 0}}, Hd[3][2][2] = {}, g[2] = {0, 0};
  for (int s = 0; s < 4; ++s) {
    const LD phi[2] = {static_cast<LD>(nb[s]), static_cast<LD>(ta[s])};
    for (int a = 0; a < 2; ++a) {
      g[a] += W[s] * phi[a] * energy[s];
      for (int b = 0; b < 2; ++b) {
        H[a][b] += W[s] * phi[a] * phi[b];
        Hd[ds[s]][a][b] += W[s] * phi[a] * phi[b];
      }
    }
  }
  const LD det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
  const LD inv[2][2] = {{H[1][1] / det, -H[0][1] / det},
                        {-H[1][0] / det, H[0][0] / det}};
  const LD b0 = inv[0][0] * g[0] + inv[0][1] * g[1];
  const LD b1 = inv[1][0] * g[0] + inv[1][1] * g[1];
  Oracle o;
  for (int d = 1; d <= 2; ++d) {
    LD tr = 0;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) tr += inv[a][b] * Hd[d][b][a];
    o.gamma[d] = tr;
  }
  o.gamma[0] = o.gamma[1] + o.gamma[2];
  std::array<LD, 3> se{}, sz{}, loss{};
  std::array<int, 3> n{};
  for (int s = 0; s < 4; ++s) {
    const LD r = energy[s] - (nb[s] * b0 + ta[s] * b1);
    const LD e_atom = 1000.0L * r / N[s];
    const LD z = r / sigma[s];
    for (int d : {0, ds[s]}) {
      se[d] += e_atom * e_atom;
      sz[d] += z * z;
      loss[d] += W[s] * r * r;
      ++n[d];
    }
  }
  for (int d = 0; d < 3; ++d) {
    o.rmse[d] = std::sqrt(se[d] / n[d]);
    o.rms_z[d] = std::sqrt(sz[d] / n[d]);
    o.share[d] = loss[d] / loss[0];
  }
  return o;
}

// ---------------------------------------------------------------------------
// The Ta fit of the invariant
// ---------------------------------------------------------------------------

/* A cubic cell of @p n Ta atoms at irregular positions (no atom at a
 * centre of symmetry: no row of the design matrix is zero), with forces
 * and a stress that are not zero. */
Structure ta_cell(std::size_t n, double a, double energy_shift) {
  Structure st;
  st.cell.load(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = 0.37 * i + 0.21 * std::sin(1.0 + i),
                 y = 0.11 * i + 0.17 * std::cos(2.0 * i) + 0.4,
                 z = 0.53 * i + 0.05 * i * i + 0.3;
    st.add_atom(Atom(Element("Ta"), std::fmod(x, a), std::fmod(y, a),
                     std::fmod(z, a), 0.3 * std::sin(1.0 + i),
                     -0.2 * std::cos(0.5 * i), 0.1 * std::sin(2.0 * i)));
  }
  st.energy = -8.0 * static_cast<double>(n) + energy_shift;
  for (int x = 0; x < 3; ++x)
    for (int y = x; y < 3; ++y) {
      st.stress(x, y) = 0.01 * (1 + x) - 0.004 * y;
      st.stress(y, x) = st.stress(x, y);
    }
  return st;
}

Context ta_context(int oalgo, const std::string &lambda) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 3 3 Ta Ta");
  c.add("CGRID2B", "2.8 3.5 4.2");
  c.add("SGRID2B", "2.0 2.0 2.0");
  c.add("TYPEMB", "DM_Dummy");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("RCUTMB", 5.0);
  c.add("DIMER", "false 0 false");
  c.add("FORCE", "true");
  c.add("STRESS", "true");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("OALGO", oalgo);
  c.add("LAMBDA", lambda);
  c.add("ESIGMA", "5");
  c.add("FSIGMA", "0.2 0.1");
  c.add("SSIGMA", "0.3");
  return c;
}

/* Three Ta cells in two datasets ({2 atoms, 3 atoms} and {4 atoms}). */
StructureDB ta_db() {
  StructureDB db;
  db.add(ta_cell(2, 3.3, 0.013));
  db.add(ta_cell(3, 3.6, -0.021));
  db.add(ta_cell(4, 4.1, 0.008));
  db.dbidx = {0, 2, 3};
  return db;
}

/* A cubic bcc cell of two Ta atoms of edge @p a: each atom is at a centre
 * of symmetry, so the rows of its six force components are zero, and the
 * cubic symmetry makes the rows of its off-diagonal stress entries zero.
 * Its reference forces and off-diagonal stresses are not zero (the noise
 * of a reference calculation), so their residuals are not either. */
Structure bcc_cell(double a, double energy_shift) {
  Structure st;
  st.cell.load(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.02, -0.015, 0.01));
  st.add_atom(Atom(Element("Ta"), 0.5 * a, 0.5 * a, 0.5 * a, -0.01, 0.025,
                   -0.02));
  st.energy = -16.0 + energy_shift;
  for (int x = 0; x < 3; ++x)
    for (int y = x; y < 3; ++y) {
      st.stress(x, y) = x == y ? 0.012 - 0.002 * x : 0.003 + 0.001 * (x + y);
      st.stress(y, x) = st.stress(x, y);
    }
  return st;
}

/* The singular values of the weighted design matrix of a fit of @p db with
 * @p c (an independent build: DesignMatrix with the row factors; the rows
 * of the structures, without the ridge rows the build appends after them,
 * which are zero before a solve), by LAPACK dgesvd. */
std::vector<double> weighted_singular_values(const Context &c,
                                             const StructureDB &db) {
  Context ctx = c;
  StructureDB d = db;
  tadah::mlip::NNFinder nnf;
  nnf.calc(d, ctx);
  tadah::models::DC_Selector dcs(ctx);
  tadah::mlip::DescriptorsCalc<> dc(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b,
                                    *dcs.cmb);
  tadah::mlip::DM_BF_Linear bf(ctx);
  tadah::mlip::DesignMatrix<tadah::mlip::DM_BF_Linear> dm(bf, ctx);
  tadah::mlip::Normaliser norm(ctx);
  dm.build(d, norm, dc, false);
  const tadah::core::phi_type &Phi = dm.getPhi();
  const std::size_t rows = dm.row_datasets().size();
  REQUIRE(Phi.rows() == rows + Phi.cols());
  for (std::size_t i = rows; i < Phi.rows(); ++i)
    for (std::size_t j = 0; j < Phi.cols(); ++j) REQUIRE(Phi(i, j) == 0.0);
  int m = static_cast<int>(rows), n = static_cast<int>(Phi.cols());
  std::vector<double> a(rows * Phi.cols());
  for (std::size_t j = 0; j < Phi.cols(); ++j)
    for (std::size_t i = 0; i < rows; ++i) a[i + rows * j] = Phi(i, j);
  char jobu = 'N', jobvt = 'N';
  int lda = m, ldu = 1, ldvt = 1, info = 0, lwork = -1;
  std::vector<double> s(static_cast<std::size_t>(std::min(m, n)));
  double u = 0, vt = 0, wq = 0;
  tadah::core::dgesvd_(&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(), &u,
                       &ldu, &vt, &ldvt, &wq, &lwork, &info);
  lwork = static_cast<int>(wq);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  tadah::core::dgesvd_(&jobu, &jobvt, &m, &n, a.data(), &lda, s.data(), &u,
                       &ldu, &vt, &ldvt, work.data(), &lwork, &info);
  REQUIRE(info == 0);
  return s;
}

/* A model of a potential's WEIGHTS, for its predictions. */
struct LoadedPotential {
  Context ctx;
  tadah::models::DC_Selector dcs;
  tadah::mlip::DescriptorsCalc<> dc;
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

/* The balance table of a fit of @p db with @p c, and the potential. */
struct Fitted {
  FitBalance balance;
  Context pot;
};

Fitted fit(const Context &c, const StructureDB &db) {
  MLIPWorkspaceManager ws;
  Context ctx = c;
  Trainer tr(ctx, ws);
  StructureDB d = db;
  tr.train(d);
  Fitted f;
  f.balance = tr.model->fit_balance();
  f.pot = tr.get_param_file();
  return f;
}

/* Every row of each quantity and dataset of @p b sums over the datasets to
 * the row of every dataset (n exactly; RMSE and RMS r/sigma by their sums
 * of squares over the entries of priority > 0, which for cells with no row
 * zero by symmetry are 1.5 per stress observation, its nine entries over
 * its six, as for n; share and gamma), and the shares of the quantities
 * and the prior's sum to 1. */
void check_sums(const FitBalance &b) {
  double share = b.prior_share;
  for (std::size_t q = 0; q < 3; ++q) {
    const FitBalanceRow *all = b.row(q, 0);
    if (!all) continue;
    share += all->share;
    std::size_t n = 0;
    double g = 0, s = 0, e2 = 0, z2 = 0;
    for (std::size_t d = 1; d <= b.datasets; ++d) {
      const FitBalanceRow &r = row_of(b, q, d);
      n += r.n;
      g += r.gamma;
      s += r.share;
      if (r.n > 0) {
        e2 += r.rmse * r.rmse * static_cast<double>(r.n);
        z2 += r.rms_z * r.rms_z * static_cast<double>(r.n);
      }
    }
    INFO("quantity " << q);
    CHECK(n == all->n);
    CHECK(g == Approx(all->gamma).epsilon(1e-12).margin(1e-14));
    CHECK(s == Approx(all->share).epsilon(1e-12).margin(1e-14));
    CHECK(std::sqrt(e2 / static_cast<double>(n)) ==
          Approx(all->rmse).epsilon(1e-12));
    CHECK(std::sqrt(z2 / static_cast<double>(n)) ==
          Approx(all->rms_z).epsilon(1e-12));
  }
  CHECK(share == Approx(1.0).epsilon(1e-12));
}

}  // namespace

/* Pins: the balance table of a fit whose data determine only the two
 * intercept columns of BIAS true (two species, every pair beyond the
 * cutoff), in two datasets with priorities (eweight 1, 2, 0.5, 1 and
 * WDBFILE 1 3), equals the solve by hand: per dataset and over both, gamma
 * = tr(H^-1 H_d), the energy RMSE in meV/atom, the RMS of r/sigma and the
 * share of the minimised sum (1e-9), n 2 per dataset; under ENORM ATOM
 * and SQRT (the RMSE stays per atom, sigma is N^q ESIGMA). At LAMBDA 0 the
 * rank is 2 and the prior has no share and no gamma; at LAMBDA 1e-3 (OALGO
 * 1, which leaves the intercepts out of the penalty) every column is
 * retained, the data still determine 2 effective parameters, and the
 * prior's gamma is the number of descriptor columns, which no observation
 * reaches, its share 0 (their coefficients are 0).
 * Before the fix: no balance table (fit_balance.h did not exist). */
TEST_CASE("Fit balance of two datasets equals the solve by hand",
          "[fit_balance]") {
  PeriodicTable::initialize();
  for (const char *enorm : {"ATOM", "SQRT"}) {
    const Oracle o = intercept_oracle(std::string(enorm) == "SQRT");
    for (const char *lambda : {"0", "1e-3"}) {
      DYNAMIC_SECTION("ENORM " << enorm << ", LAMBDA " << lambda) {
        const Fitted f = fit(intercept_context(lambda, enorm), intercept_db());
        const FitBalance &b = f.balance;
        INFO(b.none);
        REQUIRE(b.computed());
        REQUIRE(b.datasets == 2);
        const std::size_t cols = f.pot.size("WEIGHTS");
        REQUIRE(cols > 2);
        CHECK(b.coefficients == cols);
        CHECK(b.pinned == 0);
        CHECK(b.sigma[0] == "2");
        CHECK(b.row(1, 0) == nullptr);  // FORCE false: no force row
        CHECK(b.row(2, 0) == nullptr);
        for (std::size_t d = 0; d <= 2; ++d) {
          INFO("dataset " << d);
          const FitBalanceRow &r = row_of(b, 0, d);
          CHECK(r.n == (d == 0 ? 4u : 2u));
          CHECK(r.gamma == Approx(static_cast<double>(o.gamma[d])).epsilon(1e-9));
          CHECK(r.rmse == Approx(static_cast<double>(o.rmse[d])).epsilon(1e-9));
          CHECK(r.rms_z == Approx(static_cast<double>(o.rms_z[d])).epsilon(1e-9));
          CHECK(r.share == Approx(static_cast<double>(o.share[d])).epsilon(1e-9));
        }
        CHECK(row_of(b, 0, 0).gamma == Approx(2.0).epsilon(1e-10));
        if (std::string(lambda) == "0") {
          CHECK(b.retained == 2);
          CHECK(b.prior_gamma == 0.0);
          CHECK(b.prior_share == 0.0);
        } else {
          CHECK(b.retained == cols);
          CHECK(b.prior_gamma ==
                Approx(static_cast<double>(cols - 2)).epsilon(1e-10));
          CHECK(b.prior_share == 0.0);
        }
        check_sums(b);
      }
    }
  }
}

/* Pins: the effective parameters of a fit of energies, forces and stresses
 * (three Ta cells in two datasets, BIAS false) sum over the quantities to
 * sum_kept s_i^2 / (s_i^2 + lambda), s_i the singular values of the
 * weighted design matrix (an independent build and SVD), and with the
 * prior's row to the rank (3): at LAMBDA 0 and 1e-2 under OALGO 1, at
 * LAMBDA 0, 1e-2 and -1 (the lambda of the evidence approximation,
 * ALPHA/BETA of the potential) under OALGO 3; 1e-10. The shares of the
 * quantities and the prior's sum to 1, and every per-dataset row sums to
 * the row of every dataset; n is 3 energies, 27 force components and 18
 * stress entries (six per tensor); the prior has a share and a gamma with
 * lambda > 0 and none at 0. The RMSE of each quantity is that of the
 * potential's predictions in its metric's unit: the energy per atom in
 * meV/atom, the force components in eV/Å and the nine entries of each
 * stress tensor in GPa; the RMS of r/sigma that of the residuals over
 * ESIGMA, FSIGMA a b (with the structure's own force RMS) and SSIGMA.
 * Before the fix: no balance table. */
TEST_CASE("Fit balance: gamma sums to sum s^2/(s^2 + lambda), the shares to 1",
          "[fit_balance]") {
  PeriodicTable::initialize();
  const StructureDB db = ta_db();
  const std::vector<double> s = weighted_singular_values(ta_context(3, "0"), db);
  REQUIRE(s.size() == 3);
  auto invariant = [&](double lam) {
    double sum = 0;
    for (double v : s)
      if (v > 1e-8 * s[0]) sum += v * v / (v * v + lam);
    return sum;
  };
  struct Case {
    int oalgo;
    const char *lambda;
  };
  for (const Case k : {Case{1, "0"}, Case{1, "1e-2"}, Case{3, "0"},
                       Case{3, "1e-2"}, Case{3, "-1"}}) {
    DYNAMIC_SECTION("OALGO " << k.oalgo << ", LAMBDA " << k.lambda) {
      const Fitted f = fit(ta_context(k.oalgo, k.lambda), db);
      const FitBalance &b = f.balance;
      INFO(b.none);
      REQUIRE(b.computed());
      double lam = std::stod(k.lambda);
      if (lam < 0) {
        lam = f.pot.get<double>("ALPHA") / f.pot.get<double>("BETA");
        CHECK(b.lambda_used == Approx(lam).epsilon(1e-12));
        CHECK(b.lambda == -1.0);
      } else {
        CHECK(b.lambda_used == lam);
      }
      CHECK(b.retained == 3);
      double g = 0;
      for (std::size_t q = 0; q < 3; ++q) g += row_of(b, q, 0).gamma;
      CHECK(g == Approx(invariant(lam)).epsilon(1e-10));
      CHECK(g + b.prior_gamma == Approx(3.0).epsilon(1e-10));
      if (lam == 0.0) {
        CHECK(g == Approx(3.0).epsilon(1e-10));
        CHECK(b.prior_gamma == 0.0);
        CHECK(b.prior_share == 0.0);
      } else {
        CHECK(b.prior_gamma > 0.0);
        CHECK(b.prior_share > 0.0);
      }
      CHECK(row_of(b, 0, 0).n == 3);
      CHECK(row_of(b, 1, 0).n == 27);
      CHECK(row_of(b, 2, 0).n == 18);
      CHECK(b.sigma[0] == "5");
      CHECK(b.sigma[1] == "0.2 0.1");
      CHECK(b.sigma[2] == "0.3 0");
      check_sums(b);

      // The RMSE and RMS r/sigma of the potential's predictions.
      LoadedPotential loaded(f.pot);
      StructureDB d = db;
      tadah::mlip::NNFinder nnf;
      nnf.calc(d, loaded.ctx);
      const StructureDB pred = loaded.model->predict(loaded.ctx, d, loaded.dc);
      double e2 = 0, ez2 = 0, f2 = 0, fz2 = 0, s2 = 0, sz2 = 0;
      std::size_t nf = 0;
      const double es = 5e-3, ss = 0.3 * tadah::mlip::kGPaInEvPerA3;
      for (std::size_t i = 0; i < d.size(); ++i) {
        const double N = static_cast<double>(d(i).natoms());
        const double de = (d(i).energy - pred(i).energy) / N;
        e2 += 1e6 * de * de;
        ez2 += (de / es) * (de / es);
        double fr = 0;
        for (const Atom &a : d(i).atoms)
          for (int x = 0; x < 3; ++x) fr += a.force[x] * a.force[x];
        const double fs = std::sqrt(0.04 + 0.01 * fr / (3.0 * N));
        for (std::size_t a = 0; a < d(i).natoms(); ++a)
          for (int x = 0; x < 3; ++x) {
            const double df = d(i).atoms[a].force[x] - pred(i).atoms[a].force[x];
            f2 += df * df;
            fz2 += (df / fs) * (df / fs);
            ++nf;
          }
        for (int x = 0; x < 3; ++x)
          for (int y = 0; y < 3; ++y) {
            const double dsv = d(i).stress(x, y) - pred(i).stress(x, y);
            s2 += dsv * dsv / (tadah::mlip::kGPaInEvPerA3 *
                               tadah::mlip::kGPaInEvPerA3);
            sz2 += (dsv / ss) * (dsv / ss);
          }
      }
      const double ns = 9.0 * static_cast<double>(d.size());
      CHECK(row_of(b, 0, 0).rmse == Approx(std::sqrt(e2 / 3.0)).epsilon(1e-8));
      CHECK(row_of(b, 0, 0).rms_z == Approx(std::sqrt(ez2 / 3.0)).epsilon(1e-8));
      CHECK(row_of(b, 1, 0).rmse ==
            Approx(std::sqrt(f2 / static_cast<double>(nf))).epsilon(1e-8));
      CHECK(row_of(b, 1, 0).rms_z ==
            Approx(std::sqrt(fz2 / static_cast<double>(nf))).epsilon(1e-8));
      CHECK(row_of(b, 2, 0).rmse == Approx(std::sqrt(s2 / ns)).epsilon(1e-8));
      CHECK(row_of(b, 2, 0).rms_z == Approx(std::sqrt(sz2 / ns)).epsilon(1e-8));
    }
  }
}

/* Pins: the RMSE of each quantity runs over every entry of priority > 0,
 * the entries zero by symmetry included, and so is the metric of tadah
 * predict -A (Analytics::calc_e_rmse, calc_f_rmse and calc_s_rmse_gpa of
 * the potential's predictions), per dataset and, over a single dataset of
 * every structure, for the row of every dataset (1e-8); the RMS of r/sigma
 * runs over the same entries (ESIGMA 5, FSIGMA 0.2 0.1 with each
 * structure's own force RMS, SSIGMA 0.3 0.05 with its own stress RMS;
 * 1e-8). n counts the observations only: a fit of a 3-atom and a 4-atom
 * irregular cell and of a bcc cell of two atoms, whose six force
 * components and three off-diagonal stress entries are zero rows, has 21
 * force and 15 stress observations (9 and 9 in the first dataset, the
 * 3-atom cell and the bcc cell), for 27 force components and 27 stress
 * entries. OALGO 1, LAMBDA 1e-3 and OALGO 3, LAMBDA 0.
 * Before the fix: the RMSE and r/sigma ran over the observations only: the
 * bcc cell's forces and off-diagonal stresses, whose residuals are their
 * reference values, were left out, and neither was the metric. */
TEST_CASE("Fit balance: RMSE and r/sigma over every entry, as the metrics",
          "[fit_balance]") {
  PeriodicTable::initialize();
  StructureDB db;
  db.add(ta_cell(3, 3.6, -0.021));
  db.add(bcc_cell(3.3, 0.017));
  db.add(ta_cell(4, 4.1, 0.008));
  db.dbidx = {0, 2, 3};
  struct Case {
    int oalgo;
    const char *lambda;
  };
  for (const Case k : {Case{1, "1e-3"}, Case{3, "0"}}) {
    DYNAMIC_SECTION("OALGO " << k.oalgo << ", LAMBDA " << k.lambda) {
      Context c = ta_context(k.oalgo, k.lambda);
      set_key(c, "SSIGMA", "0.3 0.05");
      const Fitted f = fit(c, db);
      const FitBalance &b = f.balance;
      INFO(b.none);
      REQUIRE(b.computed());
      REQUIRE(b.datasets == 2);
      CHECK(row_of(b, 0, 0).n == 3);
      CHECK(row_of(b, 1, 0).n == 21);
      CHECK(row_of(b, 1, 1).n == 9);
      CHECK(row_of(b, 1, 2).n == 12);
      CHECK(row_of(b, 2, 0).n == 15);
      CHECK(row_of(b, 2, 1).n == 9);
      CHECK(row_of(b, 2, 2).n == 6);

      LoadedPotential loaded(f.pot);
      StructureDB d = db;
      tadah::mlip::NNFinder nnf;
      nnf.calc(d, loaded.ctx);
      const StructureDB pred = loaded.model->predict(loaded.ctx, d, loaded.dc);
      StructureDB one = d;  // every structure in one dataset
      one.dbidx = {0, d.size()};
      const tadah::mlip::Analytics per(d, pred), all(one, pred);
      const tadah::core::t_type e = per.calc_e_rmse(), fr = per.calc_f_rmse(),
                                sr = per.calc_s_rmse_gpa();
      const tadah::core::t_type e0 = all.calc_e_rmse(), f0 = all.calc_f_rmse(),
                                s0 = all.calc_s_rmse_gpa();
      CHECK(row_of(b, 0, 0).rmse == Approx(1000.0 * e0(0)).epsilon(1e-8));
      CHECK(row_of(b, 1, 0).rmse == Approx(f0(0)).epsilon(1e-8));
      CHECK(row_of(b, 2, 0).rmse == Approx(s0(0)).epsilon(1e-8));
      for (std::size_t ds = 1; ds <= 2; ++ds) {
        INFO("dataset " << ds);
        CHECK(row_of(b, 0, ds).rmse == Approx(1000.0 * e(ds - 1)).epsilon(1e-8));
        CHECK(row_of(b, 1, ds).rmse == Approx(fr(ds - 1)).epsilon(1e-8));
        CHECK(row_of(b, 2, ds).rmse == Approx(sr(ds - 1)).epsilon(1e-8));
      }

      // The RMS of residual / error scale over the same entries, by hand:
      // sigma_s = sqrt(a^2 + (b M_s)^2), M_s the RMS of the structure's own
      // reference force components (stress entries).
      const double kG = tadah::mlip::kGPaInEvPerA3;
      std::array<std::array<double, 3>, 3> z2{}, cnt{};  // [dataset][q]
      for (std::size_t i = 0; i < d.size(); ++i) {
        const std::size_t ds = i < 2 ? 1 : 2;
        const double N = static_cast<double>(d(i).natoms());
        const double ze = (d(i).energy - pred(i).energy) / N / 5e-3;
        double fm = 0, sm = 0;
        for (const Atom &a : d(i).atoms)
          for (int x = 0; x < 3; ++x) fm += a.force[x] * a.force[x];
        for (int x = 0; x < 3; ++x)
          for (int y = 0; y < 3; ++y) sm += d(i).stress(x, y) * d(i).stress(x, y);
        const double fs = std::sqrt(0.04 + 0.01 * fm / (3.0 * N));
        const double ss = std::sqrt(std::pow(0.3 * kG, 2) + 0.0025 * sm / 9.0);
        double fz = 0, sz = 0;
        for (std::size_t a = 0; a < d(i).natoms(); ++a)
          for (int x = 0; x < 3; ++x)
            fz += std::pow((d(i).atoms[a].force[x] - pred(i).atoms[a].force[x]) /
                               fs, 2);
        for (int x = 0; x < 3; ++x)
          for (int y = 0; y < 3; ++y)
            sz += std::pow((d(i).stress(x, y) - pred(i).stress(x, y)) / ss, 2);
        for (std::size_t dd : {std::size_t{0}, ds}) {
          z2[dd][0] += ze * ze;
          cnt[dd][0] += 1;
          z2[dd][1] += fz;
          cnt[dd][1] += 3.0 * N;
          z2[dd][2] += sz;
          cnt[dd][2] += 9;
        }
      }
      for (std::size_t dd = 0; dd <= 2; ++dd)
        for (std::size_t q = 0; q < 3; ++q) {
          INFO("dataset " << dd << ", quantity " << q);
          CHECK(row_of(b, q, dd).rms_z ==
                Approx(std::sqrt(z2[dd][q] / cnt[dd][q])).epsilon(1e-8));
        }
    }
  }
}

/* Pins: the prior's effective parameters, lambda tr(H^-1), at a LAMBDA
 * far below the squared singular values of the weighted design matrix
 * (1e-7, OALGO 3) equal sum_kept lambda / (s_i^2 + lambda) of an
 * independent SVD within 1e-8 relative, and are above 0; at LAMBDA 1e-7
 * under OALGO 1 (every column penalised, BIAS false) as well.
 * Before the fix: the rank less the data's effective parameters, whose
 * rounding (about 1e-15) is larger than the prior's part (about 1e-12 and
 * below): a relative error above 1e-3, and a negative prior's gamma in
 * some fits. */
TEST_CASE("Fit balance: the prior's gamma at a small LAMBDA", "[fit_balance]") {
  PeriodicTable::initialize();
  const StructureDB db = ta_db();
  const std::vector<double> s = weighted_singular_values(ta_context(3, "0"), db);
  const double lam = 1e-7;
  double prior = 0;
  for (double v : s)
    if (v > 1e-8 * s[0]) prior += lam / (v * v + lam);
  for (int oalgo : {3, 1}) {
    DYNAMIC_SECTION("OALGO " << oalgo) {
      const Fitted f = fit(ta_context(oalgo, "1e-7"), db);
      REQUIRE(f.balance.computed());
      INFO(std::scientific << std::setprecision(17) << "prior's gamma "
                           << f.balance.prior_gamma << ", reference " << prior);
      CHECK(f.balance.prior_gamma > 0.0);
      CHECK(f.balance.prior_gamma == Approx(prior).epsilon(1e-8));
    }
  }
}

/* Pins: with FIXINDEX pinning one of the three coefficients (OALGO 1,
 * LAMBDA 0) the table counts the two the solve fitted, names the pinned
 * one in its first line, and its effective parameters sum to the rank of
 * the free columns, 2. Before the fix: no balance table. */
TEST_CASE("Fit balance with a pinned coefficient", "[fit_balance]") {
  PeriodicTable::initialize();
  Context c = ta_context(1, "0");
  c.add("FIXINDEX", "2");
  c.add("FIXWEIGHT", "0.5");
  const Fitted f = fit(c, ta_db());
  const FitBalance &b = f.balance;
  INFO(b.none);
  REQUIRE(b.computed());
  CHECK(b.coefficients == 2);
  CHECK(b.pinned == 1);
  CHECK(b.retained == 2);
  double g = 0;
  for (std::size_t q = 0; q < 3; ++q) g += row_of(b, q, 0).gamma;
  CHECK(g == Approx(2.0).epsilon(1e-10));
  check_sums(b);
  const std::vector<std::string> lines = tadah::mlip::fit_balance_lines(b);
  REQUIRE(!lines.empty());
  CHECK_THAT(lines[0], Contains("2 coefficients") &&
                           Contains("FIXINDEX pins 1") && Contains("rank 2"));
}

/* Pins: a fit whose FIXINDEX pins every coefficient (FIXINDEX 1 2 3 of the
 * three, OALGO 1 and 3) solves for none, and its table is none, saying so:
 * the one line "Fit balance: none (FIXINDEX pins every coefficient: none
 * was fitted)", the line tadah hpo records for a run whose FIXINDEX pins
 * every coefficient (it skips the fit).
 * Before the fix: a table of 0 coefficients, rank 0, with the rows of the
 * quantities, where tadah hpo records none. */
TEST_CASE("Fit balance: none when FIXINDEX pins every coefficient",
          "[fit_balance]") {
  PeriodicTable::initialize();
  for (int oalgo : {1, 3}) {
    DYNAMIC_SECTION("OALGO " << oalgo) {
      Context c = ta_context(oalgo, "0");
      c.add("FIXINDEX", "1 2 3");
      c.add("FIXWEIGHT", "0.5 0.2 0.1");
      const Fitted f = fit(c, ta_db());
      CHECK(!f.balance.computed());
      CHECK(f.balance.none ==
            "FIXINDEX pins every coefficient: none was fitted");
      CHECK(tadah::mlip::fit_balance_lines(f.balance) ==
            std::vector<std::string>{"Fit balance: none (FIXINDEX pins every "
                                     "coefficient: none was fitted)"});
    }
  }
}

/* Pins: the table survives the potential: written into a potential's
 * training record after its DBFILE line (write_potential_with_record) and
 * read back (read_training_record, read_fit_balance), it has the same
 * lines, its numbers those printed (4 significant digits from 0.001 to
 * 9999, 3 in exponent form: 5e-3 relative of the computed ones), and the
 * potential's key lines parse as before (no
 * key is read from the table); so does a table whose first line and error
 * scales are wrapped (LAMBDA -1, FIXINDEX 2 and ENORM SQRT). A record
 * without a table reads none, and a table cut short is refused naming the
 * line. Before the fix: no balance table. */
TEST_CASE("Fit balance survives writing and reading the potential",
          "[fit_balance]") {
  PeriodicTable::initialize();
  Context wrapped = ta_context(3, "-1");
  wrapped.add("FIXINDEX", "2");
  wrapped.add("FIXWEIGHT", "0.5");
  wrapped.add("ENORM", "SQRT");
  const std::vector<Context> contexts = {ta_context(3, "1e-2"), wrapped};
  for (std::size_t which = 0; which < contexts.size(); ++which) {
  DYNAMIC_SECTION((which == 0 ? "LAMBDA 1e-2" : "LAMBDA -1, FIXINDEX, SQRT")) {
  const Fitted f = fit(contexts[which], ta_db());
  REQUIRE(f.balance.computed());
  tadah::mlip::TrainingRecord rec;
  rec.lines.push_back("DBFILE a.tadah b.tadah");
  tadah::mlip::add_fit_balance(rec, f.balance);
  const std::vector<std::string> lines = tadah::mlip::fit_balance_lines(f.balance);
  REQUIRE(rec.lines.size() == 1 + lines.size());

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("tadah_fit_balance_" + std::to_string(::getpid()) + ".tadah");
  {
    std::ofstream os(path);
    tadah::mlip::write_potential_with_record(os, f.pot, rec, 18);
  }
  const tadah::mlip::TrainingRecord back =
      tadah::mlip::read_training_record(path.string());
  REQUIRE(back.lines.size() == rec.lines.size());
  CHECK(back.lines[0] == "DBFILE a.tadah b.tadah");
  CHECK(tadah::mlip::fit_balance_record_lines(back) == lines);
  const auto read = tadah::mlip::read_fit_balance(back);
  REQUIRE(read.has_value());
  CHECK(tadah::mlip::fit_balance_lines(*read) == lines);
  CHECK(read->retained == f.balance.retained);
  CHECK(read->coefficients == f.balance.coefficients);
  CHECK(read->datasets == f.balance.datasets);
  CHECK(read->sigma == f.balance.sigma);
  REQUIRE(read->rows.size() == f.balance.rows.size());
  for (std::size_t i = 0; i < read->rows.size(); ++i) {
    const FitBalanceRow &a = read->rows[i], &e = f.balance.rows[i];
    INFO("row " << i);
    CHECK(a.quantity == e.quantity);
    CHECK(a.dataset == e.dataset);
    CHECK(a.n == e.n);
    CHECK(a.rmse == Approx(e.rmse).epsilon(5e-3));
    CHECK(a.rms_z == Approx(e.rms_z).epsilon(5e-3));
    CHECK(a.share == Approx(e.share).epsilon(5e-3));
    CHECK(a.gamma == Approx(e.gamma).epsilon(5e-3));
  }
  CHECK(read->prior_share == Approx(f.balance.prior_share).epsilon(5e-3));
  CHECK(read->prior_gamma == Approx(f.balance.prior_gamma).epsilon(5e-3));
  // LAMBDA as given, and the evidence approximation's to its 6 digits.
  CHECK(read->lambda == f.balance.lambda);
  CHECK(read->lambda_used ==
        Approx(f.balance.lambda_used).epsilon(which == 0 ? 1e-12 : 1e-6));

  const Context parsed = tadah::core::CoreUserConfigParser().parse(path.string());
  REQUIRE(parsed.size("WEIGHTS") == f.pot.size("WEIGHTS"));
  for (std::size_t i = 0; i < parsed.size("WEIGHTS"); ++i)
    CHECK(parsed.get<double>("WEIGHTS", i) == f.pot.get<double>("WEIGHTS", i));
  CHECK(!parsed.has_stored("DBFILE"));
  std::filesystem::remove(path);

  tadah::mlip::TrainingRecord without;
  without.lines.push_back("DBFILE a.tadah");
  CHECK(!tadah::mlip::read_fit_balance(without).has_value());
  CHECK(tadah::mlip::fit_balance_record_lines(without).empty());

  tadah::mlip::TrainingRecord cut = rec;
  cut.lines.pop_back();  // the prior's row
  try {
    (void)tadah::mlip::read_fit_balance(cut);
    FAIL("a table without its prior row was read");
  } catch (const std::runtime_error &e) {
    CHECK_THAT(e.what(), Contains("balance") && Contains("prior"));
  }
  }
  }
}

/* A table written out by hand (FitBalance): two datasets of energies and
 * forces, numbers from 9.88e-04 to 1.00e+04, a dataset without an energy
 * observation and LAMBDA 0. */
static FitBalance hand_table() {
  FitBalance b;
  b.lambda = 0.0;
  b.lambda_used = 0.0;
  b.coefficients = 9;
  b.retained = 9;
  b.datasets = 2;
  b.sigma = {"1", "0.1 0", ""};
  auto row = [&](std::size_t q, std::size_t d, std::size_t n, double rmse,
                 double rms_z, double share, double gamma) {
    FitBalanceRow r;
    r.quantity = q;
    r.dataset = d;
    r.n = n;
    r.rmse = rmse;
    r.rms_z = rms_z;
    r.share = share;
    r.gamma = gamma;
    b.rows.push_back(r);
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  row(0, 0, 100, 0.152, 1.52, 0.3411, 4.14);
  row(0, 1, 60, 0.0009876, 0.5, 5.98e-4, 2.0);
  row(0, 2, 0, nan, nan, 0.0, 0.0);
  row(1, 0, 16200, 1234.0, 9999.7, 0.6589, 4.86);
  row(1, 1, 9000, 0.01659, 0.1659, 0.6, 3.0);
  row(1, 2, 7200, 0.02, 0.2, 0.0589, 1.86);
  b.prior_share = 0.0;
  b.prior_gamma = 0.0;
  return b;
}

/* Pins: the lines of a table: the first line "Fit balance: LAMBDA <l>;
 * <p> coefficients, rank <r>", the line of the error scales of the
 * quantities with a row ("  error scales: ESIGMA 1 meV/atom; FSIGMA 0.1 0
 * eV/Å"), the column header (quantity, dataset, n, RMSE, unit, r/sigma,
 * share, gamma), one row per quantity and dataset and the prior's row; the
 * columns two blanks apart after an indent of two, as wide as their widest
 * entry, numbers to the right. A number has 4 significant digits with its
 * trailing zeros from 0.001 to 9999 (0.1520, 1.520, 4.140, 0.5000, 0.02000,
 * 1234) and 3 in exponent form otherwise (9.88e-04, 5.98e-04, and 1.00e+04
 * for 9999.7, which rounds to 10000); 0 is "0", and a row without an
 * observation has "-" for RMSE and r/sigma.
 * Before the fix: an error scale was a column of every row, "%.4g" dropped
 * the trailing zeros (0.152, 4.14), and 9999.7 printed as "1e+04". */
TEST_CASE("Fit balance lines: the layout and the numbers", "[fit_balance]") {
  const std::vector<std::string> expected = {
      "Fit balance: LAMBDA 0; 9 coefficients, rank 9",
      "  error scales: ESIGMA 1 meV/atom; FSIGMA 0.1 0 eV/Å",
      "  quantity  dataset      n      RMSE  unit       r/sigma     share  gamma",
      "  energy    all        100    0.1520  meV/atom     1.520    0.3411  4.140",
      "  energy    1           60  9.88e-04  meV/atom    0.5000  5.98e-04  2.000",
      "  energy    2            0         -  meV/atom         -         0      0",
      "  force     all      16200      1234  eV/Å      1.00e+04    0.6589  4.860",
      "  force     1         9000   0.01659  eV/Å        0.1659    0.6000  3.000",
      "  force     2         7200   0.02000  eV/Å        0.2000   0.05890  1.860",
      "  prior" + std::string(58, ' ') + "0" + std::string(6, ' ') + "0"};
  CHECK(tadah::mlip::fit_balance_lines(hand_table()) == expected);
}

/* Pins: every line of a table is at most 78 characters wide (80 in a
 * training record, after its "# "):
 *  - three datasets of energies, forces and stresses, the second of
 *    WDBFILE 1e-4 (numbers in exponent form), and error scales of 17
 *    significant digits, as tadah hpo searches them: printed to 4 (ESIGMA
 *    5.207, FSIGMA 0.1235 0.05679, SSIGMA 0.5123 0.01235), the line of the
 *    error scales wrapped after a "; ", its next line indented by four;
 *  - LAMBDA -1 with FIXINDEX 2 and ENORM SQRT: the first line names the
 *    lambda of the evidence approximation (6 digits) and ENORM SQRT and is
 *    wrapped after a "; " ("    2 coefficients (FIXINDEX pins 1 more), rank
 *    2"), and ESIGMA is in meV per sqrt(atom);
 *  - the none line of M_KRR implementation 2, wrapped at blanks, which
 *    reads back as its reason.
 * Before the fix: the error scales were a column of every row, printed to
 * 17 digits (lines of 93 characters), the first line of LAMBDA -1 with
 * FIXINDEX was 104 wide and did not name ENORM SQRT, and the none line of
 * M_KRR implementation 2 was 150 wide. */
TEST_CASE("Fit balance lines are at most 78 characters wide", "[fit_balance]") {
  PeriodicTable::initialize();
  SECTION("three datasets, error scales of 17 digits") {
    StructureDB db = ta_db();
    db.dbidx = {0, 1, 2, 3};
    Context c = ta_context(3, "1e-2");
    c.add("WDBFILE", "1 1e-4 2");
    set_key(c, "ESIGMA", "5.206733702981638");
    set_key(c, "FSIGMA", "0.1234567890123 0.0567891234567");
    set_key(c, "SSIGMA", "0.512345678901 0.0123456789012");
    const Fitted f = fit(c, db);
    INFO(f.balance.none);
    REQUIRE(f.balance.computed());
    REQUIRE(f.balance.datasets == 3);
    CHECK(f.balance.sigma[0] == "5.207");
    CHECK(f.balance.sigma[1] == "0.1235 0.05679");
    CHECK(f.balance.sigma[2] == "0.5123 0.01235");
    const std::vector<std::string> lines =
        tadah::mlip::fit_balance_lines(f.balance);
    REQUIRE(lines.size() == 1 + 2 + 1 + 3 * 4 + 1);
    CHECK_THAT(lines[0], StartsWith("Fit balance: LAMBDA 0.01; 3 coefficients"));
    CHECK(lines[1] ==
          "  error scales: ESIGMA 5.207 meV/atom; FSIGMA 0.1235 0.05679 eV/Å;");
    CHECK(lines[2] == "    SSIGMA 0.5123 0.01235 GPa");
    CHECK(lines[3].substr(0, 20) == "  quantity  dataset ");
    const char *quantity[3] = {"energy", "force", "stress"};
    const char *unit[3] = {"meV/atom", "eV/Å", "GPa"};
    bool exponent = false;
    for (std::size_t q = 0; q < 3; ++q)
      for (std::size_t d = 0; d < 4; ++d) {
        const std::string &l = lines[4 + 4 * q + d];
        INFO(l);
        CHECK_THAT(l, StartsWith(std::string("  ") + quantity[q]));
        CHECK_THAT(l, Contains(unit[q]));
        CHECK_THAT(l, Contains(d == 0 ? " all " : " " + std::to_string(d) + " "));
        exponent = exponent || l.find("e-0") != std::string::npos;
      }
    CHECK(exponent);
    CHECK_THAT(lines.back(), StartsWith("  prior"));
    check_width(lines);
  }
  SECTION("LAMBDA -1, FIXINDEX and ENORM SQRT") {
    Context c = ta_context(3, "-1");
    c.add("FIXINDEX", "2");
    c.add("FIXWEIGHT", "0.5");
    c.add("ENORM", "SQRT");
    const Fitted f = fit(c, ta_db());
    INFO(f.balance.none);
    REQUIRE(f.balance.computed());
    const double lam =
        f.pot.get<double>("ALPHA") / f.pot.get<double>("BETA");
    const std::vector<std::string> lines =
        tadah::mlip::fit_balance_lines(f.balance);
    REQUIRE(lines.size() >= 5);
    CHECK(lines[0] == "Fit balance: LAMBDA -1 (evidence approximation: " +
                          tadah::core::format_sig(lam, 6) + "); ENORM SQRT;");
    CHECK(lines[1] == "    2 coefficients (FIXINDEX pins 1 more), rank 2");
    CHECK(lines[2] ==
          "  error scales: ESIGMA 5 meV per sqrt(atom); FSIGMA 0.2 0.1 eV/Å;");
    CHECK(lines[3] == "    SSIGMA 0.3 0 GPa");
    CHECK(lines[4].substr(0, 20) == "  quantity  dataset ");
    check_width(lines);
  }
  SECTION("the none line of M_KRR implementation 2") {
    // OALGO 1, the default: implementation 2 reads no OALGO and refuses
    // another (the case had the OALGO 3 of ta_context, without effect;
    // before that refusal the case passes with OALGO 1 as well, a pin).
    Context c = ta_context(1, "1e-6");
    set_key(c, "FORCE", "false");
    set_key(c, "STRESS", "false");
    c.remove("FSIGMA");
    c.remove("SSIGMA");
    set_key(c, "ESIGMA", "1");
    c.remove("MODEL");
    c.add("MODEL", "M_KRR");
    c.add("MODEL", "Kern_RBF");
    c.add("MODEL", 2);
    c.add("MPARAMS", 1.5);
    c.add("SBASIS", 3);
    StructureDB db = ta_db();
    db.dbidx = {0, 3};
    const Fitted f = fit(c, db);
    REQUIRE(!f.balance.computed());
    const std::vector<std::string> lines =
        tadah::mlip::fit_balance_lines(f.balance);
    REQUIRE(lines.size() > 1);
    for (std::size_t k = 1; k < lines.size(); ++k)
      CHECK_THAT(lines[k], StartsWith("    "));
    CHECK(paragraph(lines, 0) == "Fit balance: none (" + f.balance.none + ")");
    check_width(lines);
    tadah::mlip::TrainingRecord rec;
    rec.lines.push_back("DBFILE a.tadah");
    tadah::mlip::add_fit_balance(rec, f.balance);
    CHECK(tadah::mlip::fit_balance_record_lines(rec) == lines);
    const auto read = tadah::mlip::read_fit_balance(rec);
    REQUIRE(read.has_value());
    CHECK(read->none == f.balance.none);
  }
}

/* Pins: the legend of the columns: one line per column, each beginning
 * with its name (quantity, dataset, n, RMSE, unit, r/sigma, share, gamma),
 * a line on what "-" means (a row without an entry of priority > 0 has no
 * RMSE and no r/sigma), then the line on what sums: the rows of every
 * dataset ("all") with the prior's, whose shares sum to 1 and whose gamma
 * sum to the rank (the rows of the datasets repeat them); every line at
 * most 78 wide. "all" pools the entries of every dataset, n counts the
 * observations, RMSE runs over every entry of priority > 0.
 * print_fit_balance prints the lines and, when asked, the legend after
 * them; the legend is asked for once per process (first_fit_balance_legend).
 * Before the fix: a line for sigma, and "gamma ... with the prior's, the
 * rank", which a table of several datasets contradicts (its gamma column
 * sums to twice the rank); "all: every dataset", which reads as a mean
 * over the datasets, and no line on "-". */
TEST_CASE("Fit balance legend", "[fit_balance]") {
  const std::vector<std::string> legend = tadah::mlip::fit_balance_legend();
  REQUIRE(legend.size() == 10);
  std::size_t i = 0;
  for (const char *col : {"quantity", "dataset", "n", "RMSE", "unit",
                          "r/sigma", "share", "gamma", "-"}) {
    INFO(legend[i]);
    CHECK_THAT(legend[i], StartsWith(std::string("  ") + col + " "));
    ++i;
  }
  CHECK_THAT(legend[1], Contains("all: every dataset's entries together"));
  CHECK_THAT(legend[2], Contains("observations"));
  CHECK_THAT(legend[3], Contains("every entry of priority > 0"));
  CHECK_THAT(legend[8], Contains("RMSE and r/sigma") &&
                            Contains("without an entry of priority > 0"));
  CHECK_THAT(legend[9], Contains("all") && Contains("prior") &&
                            Contains("share sums to 1") &&
                            Contains("gamma to the rank"));
  check_width(legend);

  const std::vector<std::string> lines =
      tadah::mlip::fit_balance_lines(hand_table());
  std::ostringstream with, without;
  tadah::mlip::print_fit_balance(with, lines, true);
  tadah::mlip::print_fit_balance(without, lines, false);
  std::string expect;
  for (const std::string &l : lines) expect += l + "\n";
  CHECK(without.str() == expect);
  for (const std::string &l : legend) expect += l + "\n";
  CHECK(with.str() == expect);
  (void)tadah::mlip::first_fit_balance_legend();
  CHECK(!tadah::mlip::first_fit_balance_legend());
}

/* Pins: a model without a fit of its own (a Trainer that did not train)
 * and a fit of M_KRR implementation 2, which solves a kernel matrix of the
 * mean descriptors of drawn structures, have no table: "Fit balance: none
 * (...)" that names why, wrapped at blanks to lines of at most 78
 * characters, which the record keeps and reads back. Before the fix: no
 * balance table. */
namespace {

/* A 2-atom bcc cell of edge @p a sheared by @p eps (first cell vector (a,
 * eps a, 0), the second atom moved with it): its off-diagonal xy stress row
 * is of the order of eps of the largest value of its column, a genuine row
 * however small. */
Structure sheared_cell(double a, double energy_shift, double eps) {
  Structure st;
  st.cell.load(a, eps * a, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  const double h = 0.5 * a;
  st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.02, -0.015, 0.01));
  st.add_atom(Atom(Element("Ta"), h + eps * h, h, h, -0.01, 0.025, -0.02));
  st.energy = -16.0 + energy_shift;
  const double s[3][3] = {{0.012, 0.004, 0.005},
                          {0.004, 0.010, 0.006},
                          {0.005, 0.006, 0.008}};
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) st.stress(x, y) = s[x][y];
  return st;
}

/* A compressed 54-atom bcc supercell (3 x 3 x 3 cells of edge 2.9 Å) with
 * every atom moved off its site, of priority @p p for its energy, forces
 * and stress: its rows are far larger than those of a 2-atom cell. */
Structure compressed_supercell(double p) {
  Structure st;
  const double a = 2.9, L = 3 * a;
  st.cell.load(L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L);
  int k = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int l = 0; l < 3; ++l)
        for (double b : {0.0, 0.5}) {
          st.add_atom(Atom(Element("Ta"),
                           (i + b) * a + 0.05 * std::sin(k),
                           (j + b) * a + 0.04 * std::cos(2.0 * k),
                           (l + b) * a + 0.03 * std::sin(3.0 * k + 1.0),
                           0.1 * std::sin(k), 0.1 * std::cos(k), 0.05));
          ++k;
        }
  st.energy = -8.0 * 54 + 40.0;
  const double s[3][3] = {{0.5, 0.01, 0.02}, {0.01, 0.4, 0.03},
                          {0.02, 0.03, 0.45}};
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) st.stress(x, y) = s[x][y];
  st.eweight = st.fweight = st.sweight = p;
  return st;
}

}  // namespace

/* Pins: a dataset that the fit leaves out (WDBFILE 0) and one whose every
 * structure has priority 0 change nothing of a LAMBDA -1 fit: the
 * observations of the balance table (n of every row), the lambda of the
 * evidence approximation, which counts them, and the coefficients (to
 * 1e-10 relative: the solve has more rows, all zero) are those of the fit
 * without that dataset. The fit: three 2-atom cells sheared by 1e-9 with
 * FORCE and STRESS true; their xy stress rows are observations (premise:
 * four stress observations per cell). The dataset left out: a compressed
 * 54-atom supercell, whose rows are far larger.
 * Before the fix: the rows of the dataset left out set the largest values
 * of the columns, so the xy stress rows of the sheared cells fell below
 * 1e-10 of them and were not counted (stress n 9, not 12), and the
 * evidence approximation found another lambda. */
TEST_CASE("Fit balance: a dataset left out of the fit changes neither the "
          "observations nor a LAMBDA -1 fit",
          "[fit_balance][observations]") {
  PeriodicTable::initialize();
  const double eps = 1e-9;
  StructureDB a;
  a.add(sheared_cell(3.2, 0.0, eps));
  a.add(sheared_cell(3.3, 0.01, eps));
  a.add(sheared_cell(3.4, 0.02, eps));
  a.dbidx = {0, 3};
  const Context c = ta_context(3, "-1");
  const Fitted alone = fit(c, a);
  REQUIRE(alone.balance.computed());
  REQUIRE(row_of(alone.balance, 2, 0).n == 4 * a.size());

  struct Variant {
    const char *name;
    double priority;
    const char *wdbfile;
  };
  for (const Variant &v : {Variant{"WDBFILE 0", 1.0, "1 0"},
                           Variant{"priority 0", 0.0, "1 1"}}) {
    DYNAMIC_SECTION("the supercell in a dataset of " << v.name) {
      StructureDB ab = a;
      ab.add(compressed_supercell(v.priority));
      ab.dbidx = {0, 3, 4};
      Context cc = c;
      set_key(cc, "WDBFILE", v.wdbfile);
      const Fitted both = fit(cc, ab);
      REQUIRE(both.balance.computed());
      for (std::size_t q = 0; q < 3; ++q) {
        INFO("quantity " << q);
        CHECK(row_of(both.balance, q, 0).n == row_of(alone.balance, q, 0).n);
        CHECK(row_of(both.balance, q, 1).n == row_of(alone.balance, q, 0).n);
        CHECK(row_of(both.balance, q, 2).n == 0);
      }
      CHECK(both.balance.lambda_used ==
            Approx(alone.balance.lambda_used).epsilon(1e-10));
      const std::vector<std::string> &w0 = alone.pot("WEIGHTS");
      const std::vector<std::string> &w1 = both.pot("WEIGHTS");
      REQUIRE(w1.size() == w0.size());
      for (std::size_t k = 0; k < w0.size(); ++k) {
        INFO("WEIGHTS " << k);
        CHECK(std::stod(w1[k]) == Approx(std::stod(w0[k])).epsilon(1e-10));
      }
    }
  }
}

TEST_CASE("Fit balance: none without a fit of observations", "[fit_balance]") {
  PeriodicTable::initialize();
  {
    MLIPWorkspaceManager ws;
    Context c = ta_context(3, "0");
    Trainer tr(c, ws);
    const FitBalance b = tr.model->fit_balance();
    CHECK(!b.computed());
    CHECK_THAT(b.none, Contains("no fit"));
    const std::vector<std::string> lines = tadah::mlip::fit_balance_lines(b);
    CHECK(paragraph(lines, 0) == "Fit balance: none (" + b.none + ")");
    check_width(lines);
  }
  // OALGO 1, the default: implementation 2 reads no OALGO and refuses
  // another (the case had the OALGO 3 of ta_context, without effect;
  // before that refusal the case passes with OALGO 1 as well, a pin).
  Context c = ta_context(1, "1e-6");
  set_key(c, "FORCE", "false");
  set_key(c, "STRESS", "false");
  c.remove("FSIGMA");
  c.remove("SSIGMA");
  set_key(c, "ESIGMA", "1");
  c.remove("MODEL");
  c.add("MODEL", "M_KRR");
  c.add("MODEL", "Kern_RBF");
  c.add("MODEL", 2);
  c.add("MPARAMS", 1.5);
  c.add("SBASIS", 3);
  StructureDB db = ta_db();
  db.dbidx = {0, 3};
  const Fitted f = fit(c, db);
  INFO(f.balance.none);
  CHECK(!f.balance.computed());
  CHECK_THAT(f.balance.none, Contains("MODEL M_KRR Kern_RBF 2"));
  const std::vector<std::string> lines = tadah::mlip::fit_balance_lines(f.balance);
  CHECK(paragraph(lines, 0) == "Fit balance: none (" + f.balance.none + ")");
  check_width(lines);
  tadah::mlip::TrainingRecord rec;
  tadah::mlip::add_fit_balance(rec, f.balance);
  const auto read = tadah::mlip::read_fit_balance(rec);
  REQUIRE(read.has_value());
  CHECK(read->none == f.balance.none);
}
