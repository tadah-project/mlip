// test_krr_implementation2_weights.cpp
//
// M_KRR implementation 2 (MODEL M_KRR <kernel> 2) solves (K + LAMBDA I) w
// = t on the mean descriptors of whole structures and their energies per
// atom: it fits no forces and no stresses, and no weight of the data
// enters its solve. A configuration that asks for either is refused before
// the fit, naming the key (or the structure's eweight) and implementation
// 1, which fits them: FORCE true, STRESS true, an eweight other than 1,
// WDBFILE values other than 1, EWEIGHT, FWEIGHT or SWEIGHT other than 1,
// WDBFILE_AUTO other than 0 and EWEIGHT_TEMP. Before the fix each was
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

#include <array>
#include <cmath>
#include <limits>
#include <string>
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
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
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

/* Pins: each of FORCE true, STRESS true, WDBFILE 2, EWEIGHT 2, FWEIGHT 2,
 * SWEIGHT 2, WDBFILE_AUTO 0.5 and EWEIGHT_TEMP 300 is refused naming the
 * key and its value and MODEL M_KRR Kern_RBF 1 (implementation 1); an
 * eweight of 0.5 on the second structure is refused naming it; the
 * configuration with every weight 1 (EWEIGHT 1.0, WDBFILE 1, WDBFILE_AUTO
 * 0) fits.
 * Before the fix: every one of them fitted, as if it were absent. */
TEST_CASE("M_KRR implementation 2 refuses the weights and the forces and "
          "stresses it would ignore",
          "[krr][train2][refusal]") {
  PeriodicTable::initialize();
  const std::vector<std::pair<std::string, std::string>> keys = {
      {"FORCE", "true"},        {"STRESS", "true"},     {"WDBFILE", "2"},
      {"EWEIGHT", "2"},         {"FWEIGHT", "2"},       {"SWEIGHT", "2"},
      {"WDBFILE_AUTO", "0.5"},  {"EWEIGHT_TEMP", "300"}};
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
  SECTION("every weight 1 (a pin)") {
    Context c = make_context();
    set_key(c, "WDBFILE", "1");
    set_key(c, "WDBFILE_AUTO", "0");
    CHECK(fit(c, training_db()) == "no exception");
  }
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
      {"FORCE", "true"},        {"STRESS", "true"},     {"WDBFILE", "2"},
      {"EWEIGHT", "2"},         {"FWEIGHT", "0.5"},     {"SWEIGHT", "2"},
      {"WDBFILE_AUTO", "0.5"},  {"EWEIGHT_TEMP", "300"}};
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
