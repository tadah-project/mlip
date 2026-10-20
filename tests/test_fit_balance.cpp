// test_fit_balance.cpp
//
// The balance table of a fit (fit_balance.h): per quantity and per dataset
// the observations n of the fit, the error scale, the RMSE in the unit of
// the quantity's metric (meV/atom, eV/Å, GPa over the nine entries of a
// stress tensor), the root mean square of residual / error scale, the share
// of the sum the fit minimises and the effective parameters gamma, with the
// prior's row (the LAMBDA penalty and the rest of the rank).
//
// The references are independent of the code under test: a solve by hand
// of a fit whose only coefficients the data determine are the intercepts of
// two species (2 x 2 algebra in long double), an SVD of the weighted design
// matrix (LAPACK dgesvd) for the sum of the effective parameters, sum_kept
// s_i^2 / (s_i^2 + lambda), and the predictions of the trained model for
// the RMSE of each quantity in its metric's unit.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/lapack.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
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

/* The singular values of the weighted design matrix of a fit of @p db with
 * @p c (an independent build: DesignMatrix with the row factors, no ridge
 * rows), by LAPACK dgesvd. */
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
  dm.augment = false;
  tadah::mlip::Normaliser norm(ctx);
  dm.build(d, norm, dc, false);
  const tadah::core::phi_type &Phi = dm.getPhi();
  int m = static_cast<int>(Phi.rows()), n = static_cast<int>(Phi.cols());
  std::vector<double> a(Phi.rows() * Phi.cols());
  for (std::size_t j = 0; j < Phi.cols(); ++j)
    for (std::size_t i = 0; i < Phi.rows(); ++i) a[i + Phi.rows() * j] = Phi(i, j);
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
 * of squares over the observations, which are 1.5 per stress observation
 * for each of its nine entries, as for n; share and gamma), and the shares
 * of the quantities and the prior's sum to 1. */
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

/* Pins: the table survives the potential: written into a potential's
 * training record after its DBFILE line (write_potential_with_record) and
 * read back (read_training_record, read_fit_balance), it has the same
 * lines, its numbers those printed (4 significant digits from 0.001 to
 * 9999, 3 in exponent form: 5e-3 relative of the computed ones), and the
 * potential's key lines parse as before (no
 * key is read from the table). A record without a table reads none, and a
 * table cut short is refused naming the line. Before the fix: no balance
 * table. */
TEST_CASE("Fit balance survives writing and reading the potential",
          "[fit_balance]") {
  PeriodicTable::initialize();
  const Fitted f = fit(ta_context(3, "1e-2"), ta_db());
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
  CHECK(read->lambda_used == Approx(f.balance.lambda_used).epsilon(1e-12));

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

/* Pins: the lines of a table of three quantities in three datasets and its
 * legend are at most 80 characters wide (a terminal; the documentation's
 * code blocks); the first line begins "Fit balance: LAMBDA", the column
 * header names quantity, dataset, n, sigma, RMSE, unit, r/sigma, share and
 * gamma, each row begins with its quantity (energy, force, stress) and the
 * last with prior; the unit is meV/atom, eV/Å or GPa; the legend has one
 * line per column, each beginning with its name. print_fit_balance prints
 * the lines and, when asked, the legend after them; the legend is asked
 * for once per process (first_fit_balance_legend). Before the fix: no
 * balance table. */
TEST_CASE("Fit balance lines and legend", "[fit_balance]") {
  PeriodicTable::initialize();
  StructureDB db = ta_db();
  db.dbidx = {0, 1, 2, 3};
  Context c = ta_context(3, "1e-2");
  c.add("WDBFILE", "1 0.5 2");
  const Fitted f = fit(c, db);
  REQUIRE(f.balance.computed());
  REQUIRE(f.balance.datasets == 3);
  const std::vector<std::string> lines = tadah::mlip::fit_balance_lines(f.balance);
  REQUIRE(lines.size() == 2 + 3 * 4 + 1);
  CHECK_THAT(lines[0], StartsWith("Fit balance: LAMBDA 0.01; 3 coefficients"));
  for (const char *col : {"quantity", "dataset", "n", "sigma", "RMSE", "unit",
                          "r/sigma", "share", "gamma"})
    CHECK_THAT(lines[1], Contains(col));
  const char *quantity[3] = {"energy", "force", "stress"};
  const char *unit[3] = {"meV/atom", "eV/Å", "GPa"};
  for (std::size_t q = 0; q < 3; ++q)
    for (std::size_t d = 0; d < 4; ++d) {
      const std::string &l = lines[2 + 4 * q + d];
      INFO(l);
      CHECK_THAT(l, StartsWith(std::string("  ") + quantity[q]));
      CHECK_THAT(l, Contains(unit[q]));
      CHECK_THAT(l, Contains(d == 0 ? " all " : " " + std::to_string(d) + " "));
    }
  CHECK_THAT(lines.back(), StartsWith("  prior"));
  for (const std::string &l : lines) {
    INFO(l);
    CHECK(display_width(l) <= 80);
  }
  const std::vector<std::string> legend = tadah::mlip::fit_balance_legend();
  REQUIRE(legend.size() == 9);
  std::size_t i = 0;
  for (const char *col : {"quantity", "dataset", "n", "sigma", "RMSE", "unit",
                          "r/sigma", "share", "gamma"}) {
    INFO(legend[i]);
    CHECK_THAT(legend[i], StartsWith(std::string("  ") + col + " "));
    CHECK(display_width(legend[i]) <= 80);
    ++i;
  }
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
 * mean descriptors of drawn structures, have no table: one line "Fit
 * balance: none (...)" that names why, which the record keeps and reads
 * back. Before the fix: no balance table. */
TEST_CASE("Fit balance: none without a fit of observations", "[fit_balance]") {
  PeriodicTable::initialize();
  {
    MLIPWorkspaceManager ws;
    Context c = ta_context(3, "0");
    Trainer tr(c, ws);
    const FitBalance b = tr.model->fit_balance();
    CHECK(!b.computed());
    CHECK_THAT(b.none, Contains("no fit"));
  }
  Context c = ta_context(3, "1e-6");
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
  REQUIRE(lines.size() == 1);
  CHECK(lines[0] == "Fit balance: none (" + f.balance.none + ")");
  tadah::mlip::TrainingRecord rec;
  tadah::mlip::add_fit_balance(rec, f.balance);
  const auto read = tadah::mlip::read_fit_balance(rec);
  REQUIRE(read.has_value());
  CHECK(read->none == f.balance.none);
}
