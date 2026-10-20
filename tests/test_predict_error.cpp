// test_predict_error.cpp
//
// The weight covariance of a potential and what `tadah predict --error`
// reports with it (M_BLR / M_KRR predict(ctx, stdb, dc, err)).
//
// An OALGO 3 fit writes into its potential (get_param_file)
//   WCOV    N, then the N x N covariance of the WEIGHTS, column-major, in the
//           parameterisation of the WEIGHTS (with NORM true and a linear
//           function: of the weights of the raw descriptors);
//   WNOISE  for each row type the fit observed (E energies, F forces, S
//           stresses), the noise variance of one observation of priority 1
//           in the units of the data, s^2 sigma_t^2, with sigma_t the
//           absolute error scale of the type (ESIGMA/1000 eV, the energy's
//           per atom; FSIGMA a eV/A) and s^2 = RSS / (n_obs - gamma), RSS
//           the weighted residual sum of squares, n_obs the observations
//           and gamma the effective number of weights.
// predict --error reports for each predicted value the standard deviation
// of the prediction, sqrt(phi^T WCOV phi + v), v = N^2 WNOISE_E for the
// energy of a structure of N atoms (ENORM ATOM) and WNOISE_F for a force
// component, and predicts what `tadah predict` predicts.
//
// The references are computed here in long double from the library's raw
// descriptors (DescriptorsCalc, no normalisation): the energy row of a
// structure is the sum of its atoms' descriptors, a force row the
// derivative sum of the linear model, the row weights eweight / (N
// ESIGMA)^2 and fweight / FSIGMA^2 (the rule of row_weights.h).
//
// Data: six sheared and rattled Ta cells of 3 atoms (the sixth with force
// priority 0: its 9 force rows are not observations), D2_Blip with 3
// functions, Cut_Cos at 5 A, INITMB false (DSIZE 3, or 4 with BIAS true),
// FORCE true, ESIGMA 50 meV/atom, FSIGMA 1.4 eV/A. The targets are
// arbitrary smooth numbers, so the fit has residuals.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_linear.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_linear.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/models/m_blr.h>
#include <tadah/mlip/models/m_krr.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/functions/basis_functions/bf_linear.h>
#include <tadah/models/functions/kernels/kern_linear.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::DescriptorsCalc;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
using LD = long double;
using LMat = std::vector<std::vector<LD>>;

std::uint64_t dbits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

void set_key(Context &c, const std::string &key, const std::string &value) {
  c.remove(key);
  c.add(key, value);
}

Context make_context(const std::string &model, const std::string &fn,
                     bool norm, const std::string &lambda) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 3 3 Ta Ta");
  c.add("CGRID2B", "2.8 3.5 4.2");
  c.add("SGRID2B", "2.0 2.0 2.0");
  c.add("TYPEMB", "DM_Dummy Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("RCUTMB", 5.0);
  c.add("DIMER", "false 0 false");
  c.add("FORCE", "true");
  c.add("STRESS", "false");
  c.add("ESIGMA", 50);
  c.add("FSIGMA", 1.4);
  c.add("BIAS", norm ? "true" : "false");
  c.add("NORM", norm ? "true" : "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", model);
  c.add("MODEL", fn);
  c.add("LAMBDA", lambda);
  c.add("OALGO", "3");
  return c;
}

Structure make_cell(double a, std::array<double, 3> shear,
                    const std::vector<std::array<double, 3>> &frac,
                    std::size_t s) {
  Structure st;
  st.cell.load(a, 0.0, 0.0, shear[0] * a, a, 0.0, shear[1] * a, shear[2] * a,
               a);
  for (const auto &f : frac) {
    double r[3];
    for (int k = 0; k < 3; ++k)
      r[k] = f[0] * st.cell(0, k) + f[1] * st.cell(1, k) + f[2] * st.cell(2, k);
    st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0, 0, 0));
  }
  const double ds = static_cast<double>(s);
  st.energy = -8.1 * static_cast<double>(st.natoms()) +
              0.37 * std::sin(1.3 * ds + 0.2);
  for (std::size_t i = 0; i < st.natoms(); ++i)
    for (int k = 0; k < 3; ++k)
      st.atoms[i].force[k] =
          0.25 * std::sin(0.7 * static_cast<double>(i) + 1.1 * k + 0.3 * ds);
  return st;
}

StructureDB training_db() {
  StructureDB db;
  const double a[6] = {3.30, 3.45, 3.15, 3.38, 3.22, 3.52};
  for (std::size_t s = 0; s < 6; ++s) {
    const double d = 0.01 * static_cast<double>(s);
    db.add(make_cell(a[s], {0.05 - d, 0.02 + 0.5 * d, -0.03 + d},
                     {{0.00 + d, 0.01, 0.02 - d},
                      {0.48, 0.52 - d, 0.05 + 0.5 * d},
                      {0.27 + d, 0.55, 0.51 - 0.5 * d}},
                     s));
  }
  db(5).fweight = 0.0;  // its force rows are not observations
  return db;
}

/* A trained potential loaded as `tadah predict` loads it. */
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

/* The error scales of the cases: ESIGMA 50 meV/atom (0.05 eV) and FSIGMA
 * 1.4 eV/A. */
const LD kESigma = 0.05L;
const LD kFSigma = 1.4L;

/* One row of the linear model's design matrix in raw descriptors: the
 * value it predicts is phi . w. type 0 energy, 1 force. */
struct Row {
  std::vector<LD> phi;
  LD target = 0, weight = 0;
  int type = 0;
  std::size_t natoms = 0;  // of its structure
};

/* The WNOISE of each type for the noise variance per unit row weight s2:
 * s2 sigma^2, sigma the absolute error scale in the units of the data. */
double wnoise_e(double s2) { return s2 * static_cast<double>(kESigma * kESigma); }
double wnoise_f(double s2) { return s2 * static_cast<double>(kFSigma * kFSigma); }

/* The noise variance of the observation of row @p x: N^2 WNOISE_E for the
 * energy of a structure of N atoms (ENORM ATOM: WNOISE E is per atom),
 * WNOISE_F for a force component. */
double row_noise(const Row &x, double s2) {
  const double n = static_cast<double>(x.natoms);
  return x.type == 0 ? n * n * wnoise_e(s2) : wnoise_f(s2);
}

/* The rows of every structure of @p db (neighbour lists computed), in the
 * order of the design matrix: the energy, then 3 force rows per atom, each
 * with its weight, priority / sigma^2: eweight / (N kESigma)^2 for the
 * energy (its residual per atom over ESIGMA), fweight / kFSigma^2 for a
 * force component. */
std::vector<Row> raw_rows(const StructureDB &db, DescriptorsCalc<> &dc) {
  std::vector<Row> rows;
  for (std::size_t s = 0; s < db.size(); ++s) {
    const Structure &st = db(s);
    const StDescriptors std = dc.calc(st);
    const std::size_t D = std.dim();
    Row e;
    e.phi.assign(D, 0.0L);
    for (std::size_t a = 0; a < std.naed(); ++a)
      for (std::size_t d = 0; d < D; ++d) e.phi[d] += std.get_aed(a)(d);
    e.target = st.energy;
    const LD sE = static_cast<LD>(st.natoms()) * kESigma;
    e.weight = st.eweight / (sE * sE);
    e.type = 0;
    e.natoms = st.natoms();
    rows.push_back(e);
    for (std::size_t a = 0; a < st.natoms(); ++a) {
      for (std::size_t k = 0; k < 3; ++k) {
        Row f;
        f.phi.assign(D, 0.0L);
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
          const std::size_t j = st.nn->idx[a][jj];
          const std::size_t aa = st.get_nn_iindex(a, j, jj);
          for (std::size_t d = 0; d < D; ++d)
            f.phi[d] -= std.fd[a][jj](d, k) - std.fd[j][aa](d, k);
        }
        f.target = st.atoms[a].force[k];
        f.weight = st.fweight / (kFSigma * kFSigma);
        f.type = 1;
        f.natoms = st.natoms();
        rows.push_back(f);
      }
    }
  }
  return rows;
}

LMat inverse(LMat M) {
  const std::size_t n = M.size();
  LMat I(n, std::vector<LD>(n, 0.0L));
  for (std::size_t i = 0; i < n; ++i) I[i][i] = 1.0L;
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t p = c;
    for (std::size_t r = c + 1; r < n; ++r)
      if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
    std::swap(M[c], M[p]);
    std::swap(I[c], I[p]);
    const LD d = M[c][c];
    for (std::size_t k = 0; k < n; ++k) {
      M[c][k] /= d;
      I[c][k] /= d;
    }
    for (std::size_t r = 0; r < n; ++r)
      if (r != c) {
        const LD f = M[r][c];
        for (std::size_t k = 0; k < n; ++k) {
          M[r][k] -= f * M[c][k];
          I[r][k] -= f * I[c][k];
        }
      }
  }
  return I;
}

/* The weighted ridge fit of the rows: w, s^2 = RSS/(n_obs - gamma) and the
 * covariance s^2 (Phi^T W Phi + lambda I)^-1, n_obs the rows of non-zero
 * weight. */
struct Reference {
  std::vector<LD> w;
  LMat Sigma;
  LD s2 = 0, gamma = 0, rss = 0;
  std::size_t n_obs = 0;
};

Reference reference_fit(const std::vector<Row> &rows, LD lambda) {
  const std::size_t n = rows[0].phi.size();
  LMat G(n, std::vector<LD>(n, 0.0L));
  std::vector<LD> b(n, 0.0L);
  Reference r;
  for (const Row &x : rows) {
    if (x.weight > 0) ++r.n_obs;
    for (std::size_t p = 0; p < n; ++p) {
      b[p] += x.weight * x.phi[p] * x.target;
      for (std::size_t q = 0; q < n; ++q)
        G[p][q] += x.weight * x.phi[p] * x.phi[q];
    }
  }
  LMat M = G;
  for (std::size_t p = 0; p < n; ++p) M[p][p] += lambda;
  const LMat Minv = inverse(M);
  r.w.assign(n, 0.0L);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) r.w[p] += Minv[p][q] * b[q];
  for (const Row &x : rows) {
    LD e = x.target;
    for (std::size_t p = 0; p < n; ++p) e -= x.phi[p] * r.w[p];
    r.rss += x.weight * e * e;
  }
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) r.gamma += G[p][q] * Minv[q][p];
  r.s2 = r.rss / (static_cast<LD>(r.n_obs) - r.gamma);
  r.Sigma = Minv;
  for (auto &row : r.Sigma)
    for (auto &v : row) v *= r.s2;
  return r;
}

LD quad(const LMat &S, const std::vector<LD> &phi) {
  LD v = 0;
  for (std::size_t p = 0; p < phi.size(); ++p)
    for (std::size_t q = 0; q < phi.size(); ++q) v += phi[p] * S[p][q] * phi[q];
  return v;
}

std::vector<double> stored_doubles(const Context &c, const std::string &key,
                                   std::size_t from = 0) {
  std::vector<double> v;
  if (!c.has_stored(key)) return v;
  for (std::size_t i = from; i < c.size(key); ++i)
    v.push_back(c.get<double>(key, i));
  return v;
}

/* The WNOISE value of row type @p t ("E", "F", "S"); -1 when absent. */
double wnoise_of(const Context &c, const std::string &t) {
  if (!c.has_stored("WNOISE")) return -1.0;
  for (std::size_t i = 0; i + 1 < c.size("WNOISE"); i += 2)
    if (c.get<std::string>("WNOISE", i) == t)
      return c.get<double>("WNOISE", i + 1);
  return -1.0;
}

std::string refusal_of_predict(Context pot, StructureDB &db) {
  try {
    LoadedPotential lp(pot);
    tadah::core::aed_type err;
    lp.model->predict(lp.ctx, db, lp.dc, err);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "<no exception>";
}

const std::array<std::array<const char *, 2>, 2> kLinearModels = {
    std::array<const char *, 2>{"M_BLR", "BF_Linear"},
    std::array<const char *, 2>{"M_KRR", "Kern_Linear"}};

}  // namespace

/* Pins: for M_BLR BF_Linear and M_KRR Kern_Linear trained with OALGO 3,
 * NORM false (LAMBDA 1e-3, BIAS false) and NORM true (LAMBDA 0, BIAS
 * true), on the six cells (51 rows with a non-zero weight: 6 energies and
 * 45 forces; the 9 force rows of the sixth cell, force weight 0, are not
 * observations):
 *  - the potential's WCOV is N = 3 (4) and s^2 (Phi^T W Phi + lambda I)^-1
 *    of the raw rows (1e-8 relative), s^2 = RSS/(51 - gamma), and its
 *    WEIGHTS the raw-descriptor solution (1e-8);
 *  - WNOISE holds E s^2 ESIGMA^2 and F s^2 FSIGMA^2 (ESIGMA 0.05 eV per
 *    atom, FSIGMA 1.4 eV/A; 1e-8) and no S;
 *  - predict with errors predicts what predict predicts (1e-10 relative);
 *  - each error is sqrt(phi^T WCOV phi + WNOISE_t) (1e-8 relative) and
 *    none is negative.
 * Before the fix: the potential had SIGMA (V^T R V, unscaled) and no WCOV
 * or WNOISE; with NORM true the error path normalised the descriptors and
 * applied the raw-descriptor weights to them, so its energies were off by
 * up to 0.99 eV (-23.23 against -24.22 eV for a 3-atom cell); every error
 * was sqrt(phi_i^T diag(SIGMA) phi_i) - sqrt(the mean of that over the
 * rows, augmentation rows included), so most force errors were negative
 * (-0.04 to -0.37 eV/A against 1.86 to 1.93) and none was the predictive
 * standard deviation (energy 0.62 against 1.01 eV). */
TEST_CASE("predict --error: the predictive standard deviation of the "
          "values predict predicts",
          "[predict_error]") {
  PeriodicTable::initialize();
  for (const auto &mf : kLinearModels) {
    for (const bool norm : {false, true}) {
      DYNAMIC_SECTION(mf[0] << " " << mf[1] << ", NORM " << norm) {
        const std::string lambda = norm ? "0" : "1e-3";
        Context c = make_context(mf[0], mf[1], norm, lambda);
        MLIPWorkspaceManager ws;
        Trainer tr(c, ws);
        StructureDB db = training_db();
        tr.train(db);
        const Context pot = tr.get_param_file();

        const std::vector<Row> rows = raw_rows(db, tr.dc);
        const Reference ref = reference_fit(rows, std::stold(lambda));
        REQUIRE(ref.n_obs == 51);
        const std::size_t N = norm ? 4 : 3;
        REQUIRE(ref.w.size() == N);

        const std::vector<double> w = stored_doubles(pot, "WEIGHTS");
        REQUIRE(w.size() == N);
        for (std::size_t p = 0; p < N; ++p)
          CHECK(std::fabs(w[p] - static_cast<double>(ref.w[p])) <
                1e-8 * std::fabs(static_cast<double>(ref.w[p])));

        const std::vector<double> wcov = stored_doubles(pot, "WCOV");
        CHECK(wcov.size() == N * N + 1);
        if (wcov.size() == N * N + 1) {
          CHECK(wcov[0] == static_cast<double>(N));
          LD dmax = 0, rmax = 0;
          for (std::size_t q = 0; q < N; ++q)
            for (std::size_t p = 0; p < N; ++p) {
              dmax = std::max(dmax, std::fabs(wcov[1 + N * q + p] - ref.Sigma[p][q]));
              rmax = std::max(rmax, std::fabs(ref.Sigma[p][q]));
            }
          CHECK(static_cast<double>(dmax / rmax) < 1e-8);
        }
        const double s2 = static_cast<double>(ref.s2);
        CHECK(wnoise_of(pot, "E") == Approx(wnoise_e(s2)).epsilon(1e-8));
        CHECK(wnoise_of(pot, "F") == Approx(wnoise_f(s2)).epsilon(1e-8));
        CHECK(wnoise_of(pot, "S") == -1.0);

        LoadedPotential lp(pot);
        tadah::core::aed_type err;
        const StructureDB with_err = lp.model->predict(lp.ctx, db, lp.dc, err);
        const StructureDB plain = lp.model->predict(lp.ctx, db, lp.dc);
        REQUIRE(with_err.size() == db.size());
        REQUIRE(plain.size() == db.size());
        REQUIRE(err.size() >= rows.size());
        std::size_t i = 0, negative = 0;
        for (std::size_t s = 0; s < db.size(); ++s) {
          INFO("structure " << s);
          CHECK(with_err(s).energy ==
                Approx(plain(s).energy).epsilon(1e-10).margin(1e-12));
          for (std::size_t a = 0; a < db(s).natoms(); ++a)
            for (std::size_t k = 0; k < 3; ++k)
              CHECK(with_err(s).atoms[a].force[k] ==
                    Approx(plain(s).atoms[a].force[k]).epsilon(1e-10).margin(1e-12));
        }
        for (const Row &x : rows) {
          INFO("row " << i << " (type " << x.type << ")");
          const double noise = row_noise(x, s2);
          const double expected =
              std::sqrt(static_cast<double>(quad(ref.Sigma, x.phi)) + noise);
          CHECK(err(i) == Approx(expected).epsilon(1e-8));
          if (!(err(i) >= 0.0)) ++negative;
          ++i;
        }
        CHECK(negative == 0);
      }
    }
  }
}

/* Pins: a model trained three times in place (Trainer::train, then
 * model->reset_trained() and Trainer::train again, as a host that retrains
 * its model does) writes one WCOV of N*N+1 = 10 values and one WNOISE (E
 * and F), equal bit for bit to those of a fresh Trainer's fit, for M_BLR
 * BF_Linear and M_KRR Kern_Linear (OALGO 3, LAMBDA 1e-3).
 * Before the fix: the potentials had no WCOV and no WNOISE (the SIGMA in
 * the model's Context was replaced on each fit). */
TEST_CASE("A model trained in place three times writes one WCOV, that of "
          "its last fit",
          "[predict_error][retrain]") {
  PeriodicTable::initialize();
  for (const auto &mf : kLinearModels) {
    DYNAMIC_SECTION(mf[0] << " " << mf[1]) {
      Context c = make_context(mf[0], mf[1], false, "1e-3");
      Context fresh_pot;
      {
        MLIPWorkspaceManager ws;
        Trainer tr(c, ws);
        StructureDB db = training_db();
        tr.train(db);
        fresh_pot = tr.get_param_file();
      }
      MLIPWorkspaceManager ws;
      Trainer tr(c, ws);
      StructureDB db = training_db();
      for (int fit = 0; fit < 3; ++fit) {
        if (fit) tr.model->reset_trained();
        tr.train(db);
      }
      const Context pot = tr.get_param_file();
      const std::vector<double> wcov = stored_doubles(pot, "WCOV");
      const std::vector<double> ref = stored_doubles(fresh_pot, "WCOV");
      REQUIRE(wcov.size() == 10);
      REQUIRE(ref.size() == 10);
      for (std::size_t k = 0; k < 10; ++k) CHECK(dbits(wcov[k]) == dbits(ref[k]));
      REQUIRE(pot.size("WNOISE") == 4);
      CHECK(dbits(wnoise_of(pot, "E")) == dbits(wnoise_of(fresh_pot, "E")));
      CHECK(dbits(wnoise_of(pot, "F")) == dbits(wnoise_of(fresh_pot, "F")));
    }
  }
}

/* Pins: a model built directly from a Context that carries the covariance
 * of another fit (SIGMA, WCOV and WNOISE, as a potential used as the
 * configuration carries them), trained with OALGO 1, writes a potential
 * with none of the three, for M_BLR BF_Linear and M_KRR Kern_Linear.
 * Before the fix: WCOV and WNOISE were not keys (Context::add threw "Key
 * not allowed: WCOV"); get_param_file copied the model's Context but for
 * WEIGHTS, NMEAN and NSTDEV, so a key of another fit stayed (the fit
 * removed SIGMA). */
TEST_CASE("A fit without a covariance writes none, whatever its Context "
          "carries",
          "[predict_error][retrain]") {
  PeriodicTable::initialize();
  for (const auto &mf : kLinearModels) {
    DYNAMIC_SECTION(mf[0] << " " << mf[1]) {
      Context c = make_context(mf[0], mf[1], false, "1e-3");
      set_key(c, "OALGO", "1");
      c.add("SIGMA", "2 1 0 0 1");
      c.add("WCOV", "2 1 0 0 1");
      c.add("WNOISE", "E 0.5 F 0.25");
      MLIPWorkspaceManager ws;
      tadah::models::DC_Selector dcs(c);
      DescriptorsCalc<> dc(c, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb);
      StructureDB db = training_db();
      tadah::mlip::NNFinder nnf;
      nnf.calc(db, c);
      Context pot;
      if (std::string(mf[0]) == "M_BLR") {
        tadah::mlip::DM_BF_Linear bf(c);
        tadah::mlip::M_BLR<tadah::mlip::DM_BF_Linear> m(bf, c);
        m.train(db, dc);
        pot = m.get_param_file();
      } else {
        tadah::mlip::DM_Kern_Linear k(c);
        tadah::mlip::M_KRR<tadah::mlip::DM_Kern_Linear> m(k, c);
        m.train(db, dc);
        pot = m.get_param_file();
      }
      CHECK(pot.size("WEIGHTS") == 3);
      CHECK_FALSE(pot.has_stored("SIGMA"));
      CHECK_FALSE(pot.has_stored("WCOV"));
      CHECK_FALSE(pot.has_stored("WNOISE"));
    }
  }
}

/* Pins: OALGO 3 with FIXINDEX 1-3 (every weight pinned, nothing solved):
 * the potential has WCOV 3 and nine zeros (no weight has a variance) and
 * WNOISE E s^2 ESIGMA^2 and F s^2 FSIGMA^2 with s^2 = RSS/51 (s^2 =
 * RSS/n_obs: no weight is fitted), RSS the weighted residuals of the pinned
 * weights (1e-10 relative); predict with errors gives the square root of
 * the noise of each value's observation (N^2 WNOISE E for an energy,
 * WNOISE F for a force component) for every value.
 * Before the fix: no WCOV and no WNOISE (and no SIGMA). */
TEST_CASE("predict --error with every weight pinned: the noise alone",
          "[predict_error][fixed_weights]") {
  PeriodicTable::initialize();
  Context c = make_context("M_BLR", "BF_Linear", false, "0");
  c.add("FIXINDEX", "1-3");
  c.add("FIXWEIGHT", "0.25 -0.5 0.75");
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  StructureDB db = training_db();
  tr.train(db);
  const Context pot = tr.get_param_file();
  const std::vector<Row> rows = raw_rows(db, tr.dc);
  const LD w_pin[3] = {0.25L, -0.5L, 0.75L};
  LD rss = 0;
  for (const Row &x : rows) {
    LD e = x.target;
    for (std::size_t p = 0; p < 3; ++p) e -= x.phi[p] * w_pin[p];
    rss += x.weight * e * e;
  }
  const double s2 = static_cast<double>(rss / 51.0L);
  const std::vector<double> wcov = stored_doubles(pot, "WCOV");
  CHECK(wcov.size() == 10);
  if (wcov.size() == 10) {
    CHECK(wcov[0] == 3.0);
    for (std::size_t k = 1; k < 10; ++k) CHECK(wcov[k] == 0.0);
  }
  CHECK(wnoise_of(pot, "E") == Approx(wnoise_e(s2)).epsilon(1e-10));
  CHECK(wnoise_of(pot, "F") == Approx(wnoise_f(s2)).epsilon(1e-10));
  if (wcov.size() == 10 && pot.has_stored("WNOISE")) {
    LoadedPotential lp(pot);
    tadah::core::aed_type err;
    lp.model->predict(lp.ctx, db, lp.dc, err);
    REQUIRE(err.size() >= rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
      CHECK(err(i) == Approx(std::sqrt(row_noise(rows[i], s2)))
                          .epsilon(1e-10));
  }
}

/* Pins: predict with errors refuses, before predicting:
 *  - a potential without WCOV and WNOISE (an OALGO 1 fit), naming WCOV and
 *    OALGO 3;
 *  - a WCOV whose N is not the number of WEIGHTS (N = 2 for 3 weights),
 *    naming WCOV, N = 2 and the 3 WEIGHTS;
 *  - a WCOV whose count is not N*N+1 (N = 3 with 8 values after it),
 *    naming WCOV, N = 3 and the 10 values it needs;
 *  - forces asked for (FORCE true) of a potential whose fit had no force
 *    rows (FORCE false: WNOISE has E only), or none of a priority above 0
 *    (every structure's fweight 0), naming the forces and WNOISE;
 * for M_BLR BF_Linear and M_KRR Kern_Linear (OALGO 3, LAMBDA 1e-3).
 * Before the fix: the first threw "Sigma matrix is not computed." (no
 * SIGMA), which names neither WCOV nor OALGO 3; with FORCE true and a fit
 * without forces (or with FWEIGHT 0, the retired global weight) the
 * errors were computed, no
 * exception; a WCOV could not be set (Context::add threw "Key not allowed:
 * WCOV"). */
TEST_CASE("predict --error refuses a covariance that does not fit the "
          "potential",
          "[predict_error]") {
  PeriodicTable::initialize();
  for (const auto &mf : kLinearModels) {
    DYNAMIC_SECTION(mf[0] << " " << mf[1]) {
      StructureDB db = training_db();
      auto train = [&](Context c, double fweight = -1.0) {
        MLIPWorkspaceManager ws;
        Trainer tr(c, ws);
        StructureDB tdb = training_db();
        if (fweight >= 0.0)
          for (std::size_t s = 0; s < tdb.size(); ++s) tdb(s).fweight = fweight;
        tr.train(tdb);
        return tr.get_param_file();
      };
      tadah::mlip::NNFinder nnf;
      Context base = make_context(mf[0], mf[1], false, "1e-3");
      nnf.calc(db, base);
      {
        Context c = base;
        set_key(c, "OALGO", "1");
        const std::string m = refusal_of_predict(train(c), db);
        INFO("OALGO 1: " << m);
        CHECK(m.find("WCOV") != std::string::npos);
        CHECK(m.find("OALGO 3") != std::string::npos);
      }
      for (const char *what : {"FORCE false", "every fweight 0"}) {
        Context c = base;
        const bool off = std::string(what) == "FORCE false";
        if (off) {
          set_key(c, "FORCE", "false");
          c.remove("FSIGMA");  // the default: FORCE false refuses another
        }
        Context p = train(c, off ? -1.0 : 0.0);
        CHECK(wnoise_of(p, "F") == -1.0);
        set_key(p, "FORCE", "true");  // predict -F
        const std::string m = refusal_of_predict(p, db);
        INFO(what << ": " << m);
        CHECK(m.find("forces") != std::string::npos);
        CHECK(m.find("WNOISE") != std::string::npos);
      }
      const Context pot = train(base);
      {
        Context p = pot;
        set_key(p, "WCOV", "2 1 0 0 1");
        const std::string m = refusal_of_predict(p, db);
        INFO("N = 2: " << m);
        CHECK(m.find("WCOV") != std::string::npos);
        CHECK(m.find("N = 2") != std::string::npos);
        CHECK(m.find("3 WEIGHTS") != std::string::npos);
      }
      {
        Context p = pot;
        set_key(p, "WCOV", "3 1 0 0 0 1 0 0 0");
        const std::string m = refusal_of_predict(p, db);
        INFO("N = 3, 9 values: " << m);
        CHECK(m.find("WCOV") != std::string::npos);
        CHECK(m.find("N = 3") != std::string::npos);
        CHECK(m.find("10 values") != std::string::npos);
      }
    }
  }
}

/* Pins: with two species (Nb, Ta), BIAS true (one intercept column per
 * species) and NORM true, M_BLR BF_Linear and M_KRR Kern_Linear (OALGO 3,
 * LAMBDA 0) write the covariance of the raw-descriptor weights: WCOV is
 * s^2 (Phi^T W Phi)^-1 of the raw rows (both intercept columns and the 3
 * D2_Blip columns, N = 5; 1e-8 relative), WNOISE E and F are s^2
 * ESIGMA^2 and s^2 FSIGMA^2 with s^2 = RSS/(51 - 5), and predict with
 * errors predicts what predict predicts with sqrt(phi^T WCOV phi + v) for
 * every row, v the noise of its observation (1e-8). The cells have 1 to 3 Nb atoms, so the energies determine both
 * intercepts.
 * Before the fix: no WCOV and no WNOISE (the potentials had SIGMA); the
 * WEIGHTS were already those of the raw descriptors (the per-species
 * intercepts of the multi-species BIAS fix), as the case checks first. */
TEST_CASE("predict --error with two species, BIAS and NORM: the covariance "
          "of the raw-descriptor weights",
          "[predict_error][bias]") {
  PeriodicTable::initialize();
  for (const auto &mf : kLinearModels) {
    DYNAMIC_SECTION(mf[0] << " " << mf[1]) {
      Context c = make_context(mf[0], mf[1], true, "0");
      set_key(c, "ATOM", "Nb Ta");
      c.remove("WATOM");
      c.add("WATOM", "1 1");
      set_key(c, "TYPE2B", "D2_Blip 3 3 * *");
      set_key(c, "TYPEMB", "DM_Dummy * *");
      StructureDB db = training_db();
      const char *species[6][3] = {{"Nb", "Ta", "Ta"}, {"Nb", "Nb", "Ta"},
                                   {"Ta", "Nb", "Ta"}, {"Nb", "Nb", "Nb"},
                                   {"Ta", "Ta", "Nb"}, {"Nb", "Ta", "Nb"}};
      for (std::size_t s = 0; s < 6; ++s)
        for (std::size_t a = 0; a < 3; ++a)
          db(s).atoms[a].Element::operator=(Element(species[s][a]));
      MLIPWorkspaceManager ws;
      Trainer tr(c, ws);
      tr.train(db);
      const Context pot = tr.get_param_file();
      const std::vector<Row> rows = raw_rows(db, tr.dc);
      const Reference ref = reference_fit(rows, 0.0L);
      REQUIRE(ref.n_obs == 51);
      REQUIRE(ref.w.size() == 5);
      const std::vector<double> w = stored_doubles(pot, "WEIGHTS");
      REQUIRE(w.size() == 5);
      for (std::size_t p = 0; p < 5; ++p)
        CHECK(w[p] == Approx(static_cast<double>(ref.w[p])).epsilon(1e-8));
      const std::vector<double> wcov = stored_doubles(pot, "WCOV");
      REQUIRE(wcov.size() == 26);
      LD dmax = 0, rmax = 0;
      for (std::size_t q = 0; q < 5; ++q)
        for (std::size_t p = 0; p < 5; ++p) {
          dmax = std::max(dmax, std::fabs(wcov[1 + 5 * q + p] - ref.Sigma[p][q]));
          rmax = std::max(rmax, std::fabs(ref.Sigma[p][q]));
        }
      CHECK(static_cast<double>(dmax / rmax) < 1e-8);
      const double s2 = static_cast<double>(ref.s2);
      CHECK(wnoise_of(pot, "E") == Approx(wnoise_e(s2)).epsilon(1e-8));
      CHECK(wnoise_of(pot, "F") == Approx(wnoise_f(s2)).epsilon(1e-8));

      LoadedPotential lp(pot);
      tadah::core::aed_type err;
      const StructureDB with_err = lp.model->predict(lp.ctx, db, lp.dc, err);
      const StructureDB plain = lp.model->predict(lp.ctx, db, lp.dc);
      for (std::size_t s = 0; s < db.size(); ++s)
        CHECK(with_err(s).energy ==
              Approx(plain(s).energy).epsilon(1e-10).margin(1e-12));
      REQUIRE(err.size() >= rows.size());
      for (std::size_t i = 0; i < rows.size(); ++i) {
        INFO("row " << i);
        const double noise = row_noise(rows[i], s2);
        CHECK(err(i) == Approx(std::sqrt(static_cast<double>(
                                   quad(ref.Sigma, rows[i].phi)) + noise))
                            .epsilon(1e-8));
      }
    }
  }
}
