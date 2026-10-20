// test_krr_implementation2_weights.cpp
//
// M_KRR implementation 2 (MODEL M_KRR <kernel> 2) solves (K + LAMBDA
// sigma^2 I) w = t on the mean descriptors of whole structures and their
// energies per atom, sigma = ESIGMA / 1000 eV: the ridge of the energy
// residuals per atom over ESIGMA, as a fit of implementation 1 weighs
// them. It fits no forces and no stresses, and no priority of the data
// enters its solve. A configuration that asks for either is refused
// before the fit, naming the key (or the structure's eweight) and
// implementation 1, which fits them: FORCE true, STRESS true, an eweight
// other than 1, WDBFILE values other than 1 and ENORM SQRT. (FSIGMA and
// SSIGMA other than the default, and ENERGY false, are refused by every
// fit without forces and stresses, before this.) Before the fix each was
// ignored: the fit was the same with or without it.
//
// Like every fit, it refuses a weight of the data that is negative or not
// finite (an fweight or sweight too, which it does not read otherwise), and
// the fit of precomputed descriptors (train(StDescriptorsDB, StructureDB)),
// which runs implementation 1, refuses MODEL M_KRR <kernel> 2.
//
// The refusals are checked on a configuration and its data without a fit
// (refuse_what_krr_implementation2_ignores; tadah hpo checks them when a
// run starts), and an implementation other than 1 and 2 is refused naming
// both (require_krr_implementation, which M_KRR::train calls).
//
// Data: three Ta cells of 3, 4 and 3 atoms, D2_Blip with 3 functions,
// Kern_RBF (MPARAMS 1.5), SBASIS 3, LAMBDA 1e-6.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/data_weights.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

Context make_context() {
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
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_KRR");
  c.add("MODEL", "Kern_RBF");
  c.add("MODEL", 2);
  c.add("MPARAMS", 1.5);
  c.add("SBASIS", 3);
  c.add("LAMBDA", 1e-6);
  c.add("OALGO", 1);
  return c;
}

void set_key(Context &c, const std::string &key, const std::string &value) {
  if (c.has_stored(key)) c.remove(key);
  c.add(key, value);
}

Structure make_cell(double a, const std::vector<std::array<double, 3>> &frac,
                    std::size_t s) {
  Structure st;
  st.label = "cell " + std::to_string(s);
  st.cell.load(a, 0.0, 0.0, 0.03 * a, a, 0.0, -0.02 * a, 0.05 * a, a);
  for (const auto &f : frac) {
    double r[3];
    for (int k = 0; k < 3; ++k)
      r[k] = f[0] * st.cell(0, k) + f[1] * st.cell(1, k) + f[2] * st.cell(2, k);
    st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0, 0, 0));
  }
  st.energy = -8.1 * static_cast<double>(st.natoms()) +
              0.37 * std::sin(1.3 * static_cast<double>(s) + 0.2);
  return st;
}

StructureDB training_db() {
  StructureDB db;
  db.add(make_cell(3.30, {{0.00, 0.01, 0.02}, {0.48, 0.52, 0.05}, {0.27, 0.55, 0.51}}, 0));
  db.add(make_cell(3.45, {{0.01, -0.02, 0.00}, {0.52, 0.47, 0.03},
                          {0.49, 0.02, 0.53}, {-0.03, 0.51, 0.48}}, 1));
  db.add(make_cell(3.15, {{0.02, -0.01, 0.00}, {0.33, 0.31, 0.36}, {0.68, 0.64, 0.70}}, 2));
  db.dbidx = {0, 3};
  return db;
}

/* What a fit of @p db in @p c throws, or "no exception". */
std::string fit(Context c, StructureDB db) {
  try {
    MLIPWorkspaceManager ws;
    Trainer tr(c, ws);
    tr.nnf.calc(db, tr.context);
    tr.model->train(db, tr.dc);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

}  // namespace

/* Pins: each of FORCE true, STRESS true, WDBFILE 2 and ENORM SQRT is
 * refused naming the key and its value and MODEL M_KRR Kern_RBF 1
 * (implementation 1); an eweight of 0.5 on the second structure is refused
 * naming it; the configuration with every priority 1 and the default error
 * scales written out (WDBFILE 1, ESIGMA 1.0, ENORM ATOM, FSIGMA 0.1 0)
 * fits, and so does ESIGMA 2, which the solve applies (the next case).
 * Before the fix: every one of them fitted, as if it were absent (EWEIGHT,
 * FWEIGHT, SWEIGHT, WDBFILE_AUTO and EWEIGHT_TEMP, retired since, too). */
TEST_CASE("M_KRR implementation 2 refuses the weights and the forces and "
          "stresses it would ignore",
          "[krr][train2][refusal]") {
  PeriodicTable::initialize();
  const std::vector<std::pair<std::string, std::string>> keys = {
      {"FORCE", "true"},  {"STRESS", "true"}, {"WDBFILE", "2"},
      {"ENORM", "SQRT"}};
  for (const auto &kv : keys) {
    DYNAMIC_SECTION(kv.first << " " << kv.second) {
      Context c = make_context();
      set_key(c, kv.first, kv.second);
      CHECK_THAT(fit(c, training_db()),
                 Contains(kv.first + " " + kv.second) &&
                     Contains("MODEL M_KRR Kern_RBF 2") &&
                     Contains("MODEL M_KRR Kern_RBF 1"));
    }
  }
  SECTION("eweight 0.5") {
    StructureDB db = training_db();
    db(1).eweight = 0.5;
    CHECK_THAT(fit(make_context(), db),
               Contains("eweight 0.5") && Contains("structure 2 of 3") &&
                   Contains("'cell 1'") && Contains("MODEL M_KRR Kern_RBF 1"));
  }
  SECTION("every priority 1, the default error scales (a pin)") {
    Context c = make_context();
    set_key(c, "WDBFILE", "1");
    set_key(c, "ESIGMA", "1.0");
    set_key(c, "ENORM", "ATOM");
    set_key(c, "FSIGMA", "0.1 0");
    CHECK(fit(c, training_db()) == "no exception");
    set_key(c, "ESIGMA", "2");
    CHECK(fit(c, training_db()) == "no exception");
  }
}

namespace {

/* The WEIGHTS of a fit of @p db in @p c (a Kern_Linear model: one per
 * descriptor component), and the mean descriptor of each structure. */
std::vector<double> linear_weights(Context c, StructureDB db,
                                   std::vector<std::vector<double>> *means) {
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  tr.nnf.calc(db, tr.context);
  if (means) {
    tadah::mlip::StDescriptorsDB st = tr.dc.calc(db);
    for (std::size_t s = 0; s < db.size(); ++s) {
      std::vector<double> m(st(s).get_aed(0).size(), 0.0);
      for (std::size_t a = 0; a < st(s).naed(); ++a)
        for (std::size_t j = 0; j < m.size(); ++j)
          m[j] += st(s).get_aed(a)[j] / static_cast<double>(st(s).naed());
      means->push_back(m);
    }
  }
  tr.model->train(db, tr.dc);
  const Context pot = tr.get_param_file();
  std::vector<double> w(pot.size("WEIGHTS"));
  pot.get("WEIGHTS", w);
  return w;
}

/* The ridge regression of @p t on the rows @p x with every residual over
 * @p sigma: the W that minimises sum_r ((t_r - x_r . W) / sigma)^2 +
 * lambda |W|^2, from its normal equations (X^T X / sigma^2 + lambda I) W =
 * X^T t / sigma^2, solved by Gaussian elimination with partial pivoting. */
std::vector<double> ridge(const std::vector<std::vector<double>> &x,
                          const std::vector<double> &t, double sigma,
                          double lambda) {
  const std::size_t n = x.front().size();
  std::vector<std::vector<double>> a(n, std::vector<double>(n + 1, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t r = 0; r < x.size(); ++r)
        a[i][j] += x[r][i] * x[r][j] / (sigma * sigma);
      if (i == j) a[i][j] += lambda;
    }
    for (std::size_t r = 0; r < x.size(); ++r)
      a[i][n] += x[r][i] * t[r] / (sigma * sigma);
  }
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t p = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (std::fabs(a[i][k]) > std::fabs(a[p][k])) p = i;
    std::swap(a[k], a[p]);
    for (std::size_t i = k + 1; i < n; ++i) {
      const double f = a[i][k] / a[k][k];
      for (std::size_t j = k; j <= n; ++j) a[i][j] -= f * a[k][j];
    }
  }
  std::vector<double> w(n, 0.0);
  for (std::size_t k = n; k-- > 0;) {
    double v = a[k][n];
    for (std::size_t j = k + 1; j < n; ++j) v -= a[k][j] * w[j];
    w[k] = v / a[k][k];
  }
  return w;
}

}  // namespace

/* Pins: implementation 2 weighs LAMBDA against the energy residuals per
 * atom over ESIGMA, as implementation 1 does: MODEL M_KRR Kern_Linear 2
 * with SBASIS 4 (the vector e0 with target 0 and the mean descriptor of
 * each of the 3 structures with its energy per atom) and LAMBDA 1e-3 has
 * the WEIGHTS of the ridge regression of those 4 rows with every residual
 * over sigma = ESIGMA / 1000 eV (the oracle above), for ESIGMA 1 (the
 * default), 250 and 1000, to 1e-6 relative (the 4 x 4 kernel matrix of 3
 * descriptor components has rank 3, so at ESIGMA 1, LAMBDA sigma^2 =
 * 1e-9, the two solves agree to about 1e-7); ESIGMA 1 and 1000 give
 * WEIGHTS that differ by more than 1e-3 relative.
 * Before the fix: the solve was (K + LAMBDA I) w = t, the ridge of sigma =
 * 1 eV/atom whatever ESIGMA (ESIGMA 1000 in effect, and a refusal of any
 * other value), so the WEIGHTS of the default ESIGMA 1 were those of the
 * oracle at 1 eV. */
TEST_CASE("M_KRR implementation 2 weighs LAMBDA against the energies per "
          "atom over ESIGMA",
          "[krr][train2][error_scales]") {
  PeriodicTable::initialize();
  Context c = make_context();
  set_key(c, "MODEL", "M_KRR Kern_Linear 2");
  c.remove("MPARAMS");
  set_key(c, "SBASIS", "4");
  set_key(c, "LAMBDA", "1e-3");
  std::vector<std::vector<double>> means;
  const StructureDB db = training_db();
  (void)linear_weights(c, db, &means);
  std::vector<std::vector<double>> rows;
  std::vector<double> t;
  rows.push_back(std::vector<double>(means.front().size(), 0.0));
  rows.back()[0] = 1.0;
  t.push_back(0.0);
  for (std::size_t s = 0; s < db.size(); ++s) {
    rows.push_back(means[s]);
    t.push_back(db(s).energy / static_cast<double>(db(s).natoms()));
  }
  for (const char *esigma : {"1", "250", "1000"}) {
    DYNAMIC_SECTION("ESIGMA " << esigma) {
      Context ce = c;
      set_key(ce, "ESIGMA", esigma);
      const std::vector<double> w = linear_weights(ce, db, nullptr);
      const std::vector<double> ref =
          ridge(rows, t, std::stod(esigma) / 1000.0, 1e-3);
      REQUIRE(w.size() == ref.size());
      double scale = 0.0;
      for (double v : ref) scale = std::max(scale, std::fabs(v));
      for (std::size_t i = 0; i < w.size(); ++i) {
        INFO("weight " << i << ": " << w[i] << " against " << ref[i]);
        CHECK(std::fabs(w[i] - ref[i]) <= 1e-6 * scale);
      }
    }
  }
  Context c1000 = c;
  set_key(c1000, "ESIGMA", "1000");
  const std::vector<double> w1 = linear_weights(c, db, nullptr);
  const std::vector<double> w1000 = linear_weights(c1000, db, nullptr);
  double diff = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < w1.size(); ++i) {
    diff = std::max(diff, std::fabs(w1[i] - w1000[i]));
    scale = std::max(scale, std::fabs(w1[i]));
  }
  CHECK(diff > 1e-3 * scale);
}

/* Pins: implementation 2 (FORCE and STRESS false) refuses fweight -1 and
 * nan and sweight -2, naming the weight, its value and the structure, as
 * implementation 1 does; a valid fweight and sweight other than 1 (0.5 and
 * 3) fit, since without forces and stresses they weight nothing, as in
 * implementation 1 with FORCE and STRESS false.
 * Before the fix: fweight -1, nan and sweight -2 fitted (no exception). */
TEST_CASE("M_KRR implementation 2 refuses negative and non-finite weights",
          "[krr][train2][refusal]") {
  PeriodicTable::initialize();
  struct Case {
    const char *name;
    double fweight, sweight;
    const char *refused;  // nullptr: fits
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Case cases[] = {{"fweight -1", -1.0, 1.0, "fweight -1"},
                        {"fweight nan", nan, 1.0, "fweight nan"},
                        {"sweight -2", 1.0, -2.0, "sweight -2"},
                        {"fweight 0.5, sweight 3 (a pin)", 0.5, 3.0, nullptr}};
  for (const Case &k : cases) {
    DYNAMIC_SECTION(k.name) {
      StructureDB db = training_db();
      db(1).fweight = k.fweight;
      db(1).sweight = k.sweight;
      const std::string msg = fit(make_context(), db);
      if (k.refused)
        CHECK_THAT(msg, Contains(k.refused) && Contains("structure 2 of 3") &&
                            Contains("'cell 1'") && Contains(">= 0"));
      else
        CHECK(msg == "no exception");
    }
  }
}

/* Pins: the fit of precomputed descriptors (train(StDescriptorsDB,
 * StructureDB)) refuses MODEL M_KRR Kern_Linear 2 (FORCE and STRESS false,
 * every weight 1), naming implementation 2, implementation 1 and
 * train(StructureDB, DescriptorsCalc); MODEL M_KRR Kern_Linear 1 fits there
 * (a pin).
 * Before the fix: it fitted implementation 1's design matrix and kept
 * MODEL M_KRR Kern_Linear 2 in the potential (no exception). */
TEST_CASE("The fit of precomputed descriptors refuses M_KRR implementation 2",
          "[krr][train2][refusal]") {
  PeriodicTable::initialize();
  auto fit_precomputed = [](Context c) {
    try {
      StructureDB db = training_db();
      MLIPWorkspaceManager ws;
      Trainer tr(c, ws);
      tr.nnf.calc(db, tr.context);
      tadah::mlip::StDescriptorsDB st = tr.dc.calc(db);
      tr.model->train(st, db);
    } catch (const std::exception &e) {
      return std::string(e.what());
    }
    return std::string("no exception");
  };
  Context c = make_context();
  set_key(c, "MODEL", "M_KRR Kern_Linear 2");
  CHECK_THAT(fit_precomputed(c),
             Contains("MODEL M_KRR Kern_Linear 2") &&
                 Contains("train(StDescriptorsDB, StructureDB)") &&
                 Contains("MODEL M_KRR Kern_Linear 1") &&
                 Contains("train(StructureDB, DescriptorsCalc)"));
  set_key(c, "MODEL", "M_KRR Kern_Linear 1");
  CHECK(fit_precomputed(c) == "no exception");
}

/* The refusals of implementation 2 are checked on the configuration and
 * the data without a fit: refuse_what_krr_implementation2_ignores throws
 * the message of the fit for each setting and for an eweight other than 1,
 * and nothing when the fit fits.
 * Before the fix: the function did not exist (the test does not compile);
 * only M_KRR::train2 made the checks. */
TEST_CASE("The refusals of M_KRR implementation 2 are checked without a fit",
          "[krr][train2][refusal]") {
  PeriodicTable::initialize();
  auto checked = [](const Context &c, const StructureDB &db) {
    try {
      tadah::mlip::refuse_what_krr_implementation2_ignores(c, db);
    } catch (const std::exception &e) {
      return std::string(e.what());
    }
    return std::string("no exception");
  };
  const std::vector<std::pair<std::string, std::string>> keys = {
      {"FORCE", "true"},  {"STRESS", "true"}, {"WDBFILE", "2"},
      {"ENORM", "SQRT"}};
  for (const auto &kv : keys) {
    DYNAMIC_SECTION(kv.first << " " << kv.second) {
      Context c = make_context();
      set_key(c, kv.first, kv.second);
      const std::string msg = checked(c, training_db());
      CHECK(msg != "no exception");
      CHECK(msg == fit(c, training_db()));
    }
  }
  SECTION("eweight 0") {
    StructureDB db = training_db();
    db(2).eweight = 0.0;
    const std::string msg = checked(make_context(), db);
    CHECK_THAT(msg, Contains("eweight 0") && Contains("structure 3 of 3"));
    CHECK(msg == fit(make_context(), db));
  }
  SECTION("every weight 1 (a pin)") {
    CHECK(checked(make_context(), training_db()) == "no exception");
  }
}

/* An M_KRR implementation other than 1 and 2 (the third value of MODEL) is
 * refused by the fit, naming the MODEL line and both implementations;
 * require_krr_implementation makes the same check without a fit, and
 * accepts implementations 1 and 2, a MODEL without the third value and
 * another model.
 * Before the fix: the fit said 'This KRR implementation does exist: 3',
 * which says the opposite and names neither the MODEL line nor the
 * implementations; require_krr_implementation did not exist. */
TEST_CASE("An M_KRR implementation other than 1 and 2 is refused, naming "
          "both",
          "[krr][refusal]") {
  PeriodicTable::initialize();
  for (const char *impl : {"3", "0"}) {
    DYNAMIC_SECTION("implementation " << impl) {
      Context c = make_context();
      set_key(c, "MODEL", std::string("M_KRR Kern_RBF ") + impl);
      const std::string msg = fit(c, training_db());
      CHECK_THAT(msg, Contains(std::string("MODEL M_KRR Kern_RBF ") + impl) &&
                          Contains("implementation 1") &&
                          Contains("implementation 2") &&
                          !Contains("does exist"));
      std::string checked = "no exception";
      try {
        tadah::mlip::require_krr_implementation(c);
      } catch (const std::exception &e) {
        checked = e.what();
      }
      CHECK(checked == msg);
    }
  }
  for (const char *model :
       {"M_KRR Kern_RBF 1", "M_KRR Kern_RBF 2", "M_KRR Kern_RBF",
        "M_BLR BF_Linear"}) {
    DYNAMIC_SECTION(model << " (a pin)") {
      Context c = make_context();
      set_key(c, "MODEL", model);
      CHECK_NOTHROW(tadah::mlip::require_krr_implementation(c));
    }
  }
}

/* The LAMBDA rules of the fit of a configuration are checked without a
 * fit: require_fit_lambda throws what the fit of M_KRR implementation 2
 * (M_KRR_Train::train2), or of M_KRR implementation 1 and M_BLR
 * (LinearRegressor::train), throws for LAMBDA -1, -2 and 0 1e-12 (with
 * OALGO 1), and nothing for LAMBDA 1e-6.
 * Before the fix: the function did not exist (the test does not
 * compile). */
TEST_CASE("The LAMBDA rules of a fit are checked without a fit",
          "[krr][lambda][refusal]") {
  PeriodicTable::initialize();
  auto checked = [](const Context &c) {
    try {
      tadah::mlip::require_fit_lambda(c);
    } catch (const std::exception &e) {
      return std::string(e.what());
    }
    return std::string("no exception");
  };
  for (const char *model :
       {"M_KRR Kern_RBF 2", "M_KRR Kern_RBF 1", "M_BLR BF_Linear"}) {
    for (const char *lambda : {"-1", "-2", "0 1e-12", "1e-6"}) {
      DYNAMIC_SECTION(model << ", LAMBDA " << lambda) {
        Context c = make_context();
        set_key(c, "MODEL", model);
        set_key(c, "LAMBDA", lambda);
        const std::string fitted = fit(c, training_db());
        CHECK(checked(c) == fitted);
        CHECK((fitted == "no exception") == (std::string(lambda) == "1e-6"));
      }
    }
  }
}
