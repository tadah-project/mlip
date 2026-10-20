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
// Data: three Ta cells of 3, 4 and 3 atoms, D2_Blip with 3 functions,
// Kern_RBF (MPARAMS 1.5), SBASIS 3, LAMBDA 1e-6.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

#include <array>
#include <cmath>
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
