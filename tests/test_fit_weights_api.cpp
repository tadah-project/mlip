// test_fit_weights_api.cpp
//
// The library's fits and the weights of the data, as the C++ API reaches
// them (M_BLR and M_KRR, through Trainer's model).
//
//  - train(StDescriptorsDB&, StructureDB), the fit of precomputed
//    descriptors, applies the row factors sqrt(priority)/sigma (the error
//    scales ESIGMA, FSIGMA, SSIGMA) as train(StructureDB, DescriptorsCalc)
//    does: the same
//    WEIGHTS, and the same observations (the rows of a non-zero weight),
//    which the weight covariance of OALGO 3 (WCOV, WNOISE) is computed with.
//  - predict(StructureDB&) returns the values the last fit predicts for the
//    structures it was fitted to, without the row factors, as
//    predict(Context, StructureDB, DescriptorsCalc) does; it reads the data
//    rows only, and refuses a StructureDB other than the fit's (other
//    counts, or the same counts with other species, cells or positions)
//    and a model without a fit of its own (built from a potential, or
//    M_KRR implementation 2, which fits no design matrix).
//  - a fit refuses a per-structure weight (eweight, fweight, sweight) that
//    is negative or not a finite number, naming the weight, the structure
//    (its index and label) and its dataset, and a negative error scale
//    (ESIGMA -1) and WDBFILE value, before
//    anything else reads the weights (with BIAS true too); a dataset file
//    with a negative weight is refused as it is read.
//
// Data: three sheared and rattled Ta cells of 3, 4 and 3 atoms, D2_Blip
// with 3 functions, Cut_Cos at 5 A, INITMB false (DSIZE 3), FORCE and
// STRESS true. The targets are arbitrary smooth numbers.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/intercepts.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
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
namespace fs = std::filesystem;

std::uint64_t bits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

Context make_context(const std::vector<std::string> &model) {
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
  c.add("STRESS", "true");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  for (const std::string &m : model) c.add("MODEL", m);
  c.add("LAMBDA", 0);
  c.add("OALGO", 3);
  return c;
}

void set_key(Context &c, const std::string &key, const std::string &value) {
  if (c.has_stored(key)) c.remove(key);
  c.add(key, value);
}

Structure make_cell(double a, std::array<double, 3> shear,
                    const std::vector<std::array<double, 3>> &frac,
                    std::size_t s) {
  Structure st;
  st.label = "cell " + std::to_string(s);
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
  for (int x = 0; x < 3; ++x)
    for (int y = x; y < 3; ++y) {
      st.stress(x, y) = 0.01 * std::cos(0.9 * x + 1.7 * y + 0.4 * ds);
      st.stress(y, x) = st.stress(x, y);
    }
  return st;
}

/* The three cells, one dataset; the weights as the cases set them. */
StructureDB training_db() {
  StructureDB db;
  db.add(make_cell(3.30, {0.05, 0.02, -0.03},
                   {{0.00, 0.01, 0.02}, {0.48, 0.52, 0.05}, {0.27, 0.55, 0.51}},
                   0));
  db.add(make_cell(3.45, {-0.04, 0.03, 0.06},
                   {{0.01, -0.02, 0.00}, {0.52, 0.47, 0.03},
                    {0.49, 0.02, 0.53}, {-0.03, 0.51, 0.48}},
                   1));
  db.add(make_cell(3.15, {0.07, -0.05, 0.02},
                   {{0.02, -0.01, 0.00}, {0.33, 0.31, 0.36}, {0.68, 0.64, 0.70}},
                   2));
  db.dbidx = {0, 3};
  return db;
}

/* The weights of a case: per structure the priorities (eweight, fweight,
 * sweight), and the error scales ESIGMA (meV/atom), FSIGMA (eV/A) and
 * SSIGMA (GPa). */
struct Weights {
  std::array<std::array<double, 3>, 3> per_structure;
  double es, fs, ss;
};

StructureDB weighted_db(const Weights &w) {
  StructureDB db = training_db();
  for (std::size_t s = 0; s < 3; ++s) {
    db(s).eweight = w.per_structure[s][0];
    db(s).fweight = w.per_structure[s][1];
    db(s).sweight = w.per_structure[s][2];
  }
  return db;
}

Context weighted_context(const std::vector<std::string> &model,
                         const Weights &w) {
  Context c = make_context(model);
  set_key(c, "ESIGMA", std::to_string(w.es));
  set_key(c, "FSIGMA", std::to_string(w.fs));
  set_key(c, "SSIGMA", std::to_string(w.ss));
  return c;
}

/* The potential of a fit of @p db in @p c: of train(StructureDB,
 * DescriptorsCalc) (precomputed false) or of train(StDescriptorsDB,
 * StructureDB) (precomputed true). */
Context fitted_potential(Context c, StructureDB db, bool precomputed) {
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  tr.nnf.calc(db, tr.context);
  if (precomputed) {
    tadah::mlip::StDescriptorsDB st = tr.dc.calc(db);
    tr.model->train(st, db);
  } else {
    tr.model->train(db, tr.dc);
  }
  return tr.get_param_file();
}

std::vector<double> values(const Context &c, const std::string &key) {
  std::vector<double> v(c.size(key));
  c.get(key, v);
  return v;
}

const std::vector<std::vector<std::string>> kLinearModels = {
    {"M_BLR", "BF_Linear"}, {"M_KRR", "Kern_Linear", "1"}};

const std::vector<Weights> kWeightCases = {
    // non-unit priorities and error scales other than the defaults
    {{{{0.5, 2.0, 3.0}, {2.0, 0.25, 1.0}, {1.5, 1.0, 0.5}}}, 300.0, 1.4, 50.0},
    // structure 0 without any priority, structure 1 without forces; the
    // default error scales
    {{{{0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 1.0}}}, 1.0, 0.1, 0.5}};

/* What @p f throws, or "no exception". */
template <class F>
std::string refusal(F f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

}  // namespace

/* Pins: for M_BLR BF_Linear and M_KRR Kern_Linear 1 (OALGO 3, LAMBDA 0),
 * with non-unit priorities and ESIGMA 300, FSIGMA 1.4, SSIGMA 50, and with
 * priorities 0 (structure 0 without any, structure 1 without forces) and
 * the default error scales, train(StDescriptorsDB, StructureDB) gives the WEIGHTS
 * of train(StructureDB, DescriptorsCalc) bit for bit, and the same weight
 * covariance (WCOV and WNOISE, whose noise variance RSS / (n_obs - gamma)
 * counts the observations): the same rows, weighted alike.
 * Before the fix: the precomputed path fitted the rows without their
 * factors (its WEIGHTS did not depend on any weight), and counted the
 * observations of the weighted rows. */
TEST_CASE("train of precomputed descriptors applies the weights of the data",
          "[fit_weights][api]") {
  PeriodicTable::initialize();
  for (const auto &model : kLinearModels) {
    for (std::size_t k = 0; k < kWeightCases.size(); ++k) {
      DYNAMIC_SECTION(model[0] << " " << model[1] << ", weights case " << k) {
        const Weights &w = kWeightCases[k];
        const Context c = weighted_context(model, w);
        const Context ref = fitted_potential(c, weighted_db(w), false);
        const Context pre = fitted_potential(c, weighted_db(w), true);
        const std::vector<double> wr = values(ref, "WEIGHTS");
        const std::vector<double> wp = values(pre, "WEIGHTS");
        REQUIRE(wr.size() == wp.size());
        for (std::size_t i = 0; i < wr.size(); ++i) {
          INFO("weight " << i << ": " << wr[i] << " vs " << wp[i]);
          CHECK(bits(wr[i]) == bits(wp[i]));
        }
        REQUIRE(ref.has_stored("WNOISE"));
        REQUIRE(pre.has_stored("WNOISE"));
        // The lines as written (WNOISE holds row-type labels and values).
        CHECK(ref("WNOISE") == pre("WNOISE"));
        CHECK(ref("WCOV") == pre("WCOV"));
      }
    }
  }
}

/* Pins: after train(StructureDB, DescriptorsCalc), predict(StructureDB&)
 * of the training structures returns the energies, forces and stresses of
 * predict(Context, StructureDB, DescriptorsCalc) to 1e-10 relative (other
 * arithmetic), one structure each, for M_BLR BF_Linear, M_KRR Kern_Linear
 * 1 and M_KRR Kern_RBF 1 (SBASIS 5) with non-unit and zero priorities and
 * ESIGMA 300, FSIGMA 1.4, SSIGMA 50, and for M_BLR BF_Linear with NORM and
 * BIAS true (its WEIGHTS act on the raw descriptors, the design matrix's
 * columns on the normalised ones).
 * Before the fix: the values were those of the weighted rows (the energy of
 * a structure with eweight 0.5 under EWEIGHT 10 x sqrt(5)), the loop read
 * the ridge rows after the data rows as further structures (writing past
 * the StructureDB it returned), and a NORM model's values were those of
 * its raw-descriptor weights on normalised descriptors. */
TEST_CASE("predict(StructureDB&) returns the unweighted values of the fit's "
          "structures",
          "[fit_weights][api][predict]") {
  PeriodicTable::initialize();
  std::vector<std::vector<std::string>> models = kLinearModels;
  models.push_back({"M_KRR", "Kern_RBF", "1"});
  for (const auto &model : models) {
    for (std::size_t k = 0; k < kWeightCases.size() + 1; ++k) {
      const bool norm = k == kWeightCases.size();
      if (norm && model[1] != "BF_Linear") continue;
      DYNAMIC_SECTION(model[0] << " " << model[1]
                               << (norm ? ", NORM and BIAS true"
                                        : ", weights case " + std::to_string(k))) {
        const Weights &w = kWeightCases[norm ? 0 : k];
        Context c = weighted_context(model, w);
        if (model[1] == "Kern_RBF") {
          c.add("MPARAMS", 1.5);
          c.add("SBASIS", 5);
        }
        if (norm) {
          set_key(c, "NORM", "true");
          set_key(c, "BIAS", "true");
          set_key(c, "OALGO", "1");
        }
        StructureDB db = weighted_db(w);
        MLIPWorkspaceManager ws;
        Trainer tr(c, ws);
        tr.nnf.calc(db, tr.context);
        tr.model->train(db, tr.dc);
        const StructureDB fit = tr.model->predict(db);
        Context pc = tr.context;
        const StructureDB ref = tr.model->predict(pc, db, tr.dc);
        REQUIRE(fit.size() == db.size());
        for (std::size_t s = 0; s < db.size(); ++s) {
          INFO("structure " << s);
          CHECK(fit(s).energy == Approx(ref(s).energy).epsilon(1e-10));
          REQUIRE(fit(s).natoms() == db(s).natoms());
          for (std::size_t a = 0; a < db(s).natoms(); ++a)
            for (int i = 0; i < 3; ++i)
              CHECK(fit(s).atoms[a].force[i] ==
                    Approx(ref(s).atoms[a].force[i]).epsilon(1e-10).margin(1e-12));
          for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y)
              CHECK(fit(s).stress(x, y) ==
                    Approx(ref(s).stress(x, y)).epsilon(1e-10).margin(1e-12));
        }
      }
    }
  }
}

/* Pins: predict(StructureDB&) refuses a StructureDB other than the fit's
 * (one structure fewer), naming both counts, and a model built from a
 * potential (WEIGHTS, no fit of its own), saying how to predict instead.
 * Before the fix: the first read the fit's rows into structures that do
 * not exist, the second read a design matrix that was never built. */
TEST_CASE("predict(StructureDB&) refuses data other than the fit's",
          "[fit_weights][api][predict]") {
  PeriodicTable::initialize();
  Context c = make_context({"M_BLR", "BF_Linear"});
  StructureDB db = training_db();
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  tr.nnf.calc(db, tr.context);
  tr.model->train(db, tr.dc);
  StructureDB fewer = db;
  fewer.structures.pop_back();
  fewer.dbidx = {0, 2};
  CHECK_THAT(refusal([&] { tr.model->predict(fewer); }),
             Contains("2 structures") && Contains("3 structures"));

  Context pot = tr.get_param_file();
  MLIPWorkspaceManager ws2;
  std::unique_ptr<tadah::mlip::DM_Function_Base> fb(
      tadah::core::factory<tadah::mlip::DM_Function_Base, const Context &>(
          pot.get<std::string>("MODEL", 1), pot));
  std::unique_ptr<tadah::mlip::M_Tadah_Base> loaded(
      tadah::core::factory<tadah::mlip::M_Tadah_Base,
                           tadah::mlip::DM_Function_Base &, Context &>(
          pot.get<std::string>("MODEL", 0), *fb, pot));
  CHECK_THAT(refusal([&] { loaded->predict(db); }),
             Contains("no fit of its own") &&
                 Contains("predict(Context, StructureDB, DescriptorsCalc)"));
}

/* Pins: predict(StructureDB&) refuses a StructureDB of the fit's shape
 * (3 structures of 3, 4 and 3 atoms, one dataset) whose structures are
 * not the fit's: the first atom of structure 2 moved by 0.2 A, the cell of
 * structure 3 3% larger, the atoms of structure 1 in another order. It
 * names the first structure that differs, its label, and what predicts
 * other structures. Data with the fit's structures and other reference
 * energies, forces and stresses (the labels a prediction does not read)
 * are its structures: their values are those of the fit's data, bit for
 * bit. M_BLR BF_Linear and M_KRR Kern_Linear 1.
 * Before the fix: the counts matched, so the three were accepted, and the
 * call returned the fit's values under the other structures' labels. */
TEST_CASE("predict(StructureDB&) refuses other structures of the fit's shape",
          "[fit_weights][api][predict]") {
  PeriodicTable::initialize();
  for (const auto &model : kLinearModels) {
    DYNAMIC_SECTION(model[0] << " " << model[1]) {
      Context c = make_context(model);
      StructureDB db = training_db();
      MLIPWorkspaceManager ws;
      Trainer tr(c, ws);
      tr.nnf.calc(db, tr.context);
      tr.model->train(db, tr.dc);
      const StructureDB fit = tr.model->predict(db);

      StructureDB moved = training_db();
      moved(1).atoms[0].position[0] += 0.2;
      CHECK_THAT(refusal([&] { tr.model->predict(moved); }),
                 Contains("structure 2 of 3") && Contains("'cell 1'") &&
                     Contains("predict(Context, StructureDB, DescriptorsCalc)"));

      StructureDB scaled = training_db();
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) scaled(2).cell(i, j) *= 1.03;
      CHECK_THAT(refusal([&] { tr.model->predict(scaled); }),
                 Contains("structure 3 of 3") && Contains("'cell 2'"));

      StructureDB reordered = training_db();
      std::swap(reordered(0).atoms[0], reordered(0).atoms[2]);
      CHECK_THAT(refusal([&] { tr.model->predict(reordered); }),
                 Contains("structure 1 of 3") && Contains("'cell 0'"));

      StructureDB relabelled = training_db();
      for (std::size_t s = 0; s < relabelled.size(); ++s) {
        relabelled(s).energy += 5.0;
        relabelled(s).atoms[0].force[1] += 1.0;
        relabelled(s).stress(0, 0) += 0.1;
      }
      StructureDB again;
      REQUIRE(refusal([&] { again = tr.model->predict(relabelled); }) ==
              "no exception");
      REQUIRE(again.size() == fit.size());
      for (std::size_t s = 0; s < fit.size(); ++s)
        CHECK(bits(again(s).energy) == bits(fit(s).energy));
    }
  }
}

/* Pins: a fit refuses a per-structure weight that is negative or not
 * finite, naming the weight and its value, the structure (1-based index
 * in the training data and its label) and the dataset; every eweight -1 is
 * refused naming structure 1, and an error scale below 0 (ESIGMA -1),
 * naming ESIGMA (the global weights, whose products with the per-structure
 * ones could cancel, are retired). M_BLR BF_Linear, train(StructureDB,
 * DescriptorsCalc).
 * Before the fix: eweight -1 on one structure and fweight nan were refused
 * as a value of the design matrix that is not finite ('a descriptor or
 * basis function gave it'), EWEIGHT -1 with every eweight -1 fitted as if
 * every weight were 1 (OALGO 3 then refused it as 'noise_of_fitted_types:
 * EWEIGHT -1.000000 with rows of non-zero weight'), and EWEIGHT -1 alone
 * was refused as a design matrix value. */
TEST_CASE("A fit refuses negative and non-finite weights",
          "[fit_weights][api][refusal]") {
  PeriodicTable::initialize();
  auto fit = [](Context c, StructureDB db) {
    return refusal([&] {
      MLIPWorkspaceManager ws;
      Trainer tr(c, ws);
      tr.nnf.calc(db, tr.context);
      tr.model->train(db, tr.dc);
    });
  };
  const Context c = make_context({"M_BLR", "BF_Linear"});
  {
    StructureDB db = training_db();
    db(1).eweight = -1.0;
    CHECK_THAT(fit(c, db), Contains("eweight -1") &&
                               Contains("structure 2 of 3") &&
                               Contains("'cell 1'") && Contains("dataset 1") &&
                               Contains(">= 0"));
  }
  {
    StructureDB db = training_db();
    db(2).fweight = std::numeric_limits<double>::quiet_NaN();
    CHECK_THAT(fit(c, db), Contains("fweight nan") &&
                               Contains("structure 3 of 3") &&
                               Contains("'cell 2'"));
  }
  {
    StructureDB db = training_db();
    db(0).sweight = std::numeric_limits<double>::infinity();
    CHECK_THAT(fit(c, db), Contains("sweight inf") &&
                               Contains("structure 1 of 3"));
  }
  {
    StructureDB db = training_db();
    for (std::size_t s = 0; s < 3; ++s) db(s).eweight = -1.0;
    CHECK_THAT(fit(c, db), Contains("eweight -1") && Contains("structure 1 of 3"));
  }
  {
    Context cn = c;
    set_key(cn, "ESIGMA", "-1");
    CHECK_THAT(fit(cn, training_db()), Contains("ESIGMA -1") && Contains("> 0"));
  }
}

/* Pins: with BIAS true (an intercept, which the energies determine) the
 * weight is refused as a weight, before the intercepts are checked:
 * ESIGMA -1 names ESIGMA (EWEIGHT -1 named EWEIGHT before the key was
 * retired) and every eweight -1 names the eweight of
 * structure 1, not the intercepts, for M_BLR BF_Linear and M_KRR
 * Kern_Linear 1, through train(StructureDB, DescriptorsCalc) and
 * train(StDescriptorsDB, StructureDB).
 * Before the fix: each was refused as 'BIAS true: the training data cannot
 * determine the per-species energy constants (the intercepts) ... no
 * training structure has an energy weight (eweight x EWEIGHT > 0)', with
 * advice about BIAS false, ESHIFT and FIXINDEX. */
TEST_CASE("A fit with BIAS true refuses a negative weight as a weight",
          "[fit_weights][api][refusal]") {
  PeriodicTable::initialize();
  for (const auto &model : kLinearModels) {
    for (const bool precomputed : {false, true}) {
      DYNAMIC_SECTION(model[0] << " " << model[1]
                               << (precomputed ? ", precomputed descriptors"
                                               : ", DescriptorsCalc")) {
        Context c = make_context(model);
        set_key(c, "BIAS", "true");
        auto fit = [&](Context cc, StructureDB db) {
          return refusal([&] { (void)fitted_potential(cc, db, precomputed); });
        };
        {
          Context cn = c;
          set_key(cn, "ESIGMA", "-1");
          const std::string msg = fit(cn, training_db());
          CHECK_THAT(msg, Contains("ESIGMA -1") && Contains("> 0") &&
                              !Contains("intercept"));
        }
        {
          StructureDB db = training_db();
          for (std::size_t s = 0; s < db.size(); ++s) db(s).eweight = -1.0;
          const std::string msg = fit(c, db);
          CHECK_THAT(msg, Contains("eweight -1") && Contains("structure 1 of 3") &&
                              !Contains("intercept"));
        }
        /* The same configuration with valid weights fits (a pin). */
        CHECK(fit(c, training_db()) == "no exception");
      }
    }
  }
}

/* Pins: the check that the energies determine the intercepts (BIAS true),
 * which tadah train and tadah hpo make before any fit, refuses an invalid
 * priority as a priority: require_intercepts_determined with WDBFILE -1,
 * and TrainEngine::run (tadah train) of a dataset file with BIAS true and
 * WDBFILE -1, name WDBFILE -1 and not the intercepts (EWEIGHT -1 before
 * the key was retired). Without WDBFILE the check passes (a pin).
 * Before the fix: both were refused as 'BIAS true: the training data cannot
 * determine the per-species energy constants (the intercepts) ...'. */
TEST_CASE("The intercept check of tadah train refuses a negative weight as "
          "a weight",
          "[fit_weights][refusal]") {
  PeriodicTable::initialize();
  Context c = make_context({"M_BLR", "BF_Linear"});
  set_key(c, "BIAS", "true");
  set_key(c, "FORCE", "false");
  set_key(c, "STRESS", "false");
  MLIPWorkspaceManager ws;
  Trainer tr(c, ws);
  CHECK(refusal([&] {
          tadah::mlip::require_intercepts_determined(training_db(), tr.context,
                                                     *tr.fb);
        }) == "no exception");
  Context cn = tr.context;
  set_key(cn, "WDBFILE", "-1");
  CHECK_THAT(refusal([&] {
               tadah::mlip::require_intercepts_determined(training_db(), cn,
                                                          *tr.fb);
             }),
             Contains("WDBFILE -1") && Contains(">= 0") &&
                 !Contains("intercept"));

  const fs::path dir = fs::temp_directory_path() /
                       ("tadah_fit_weights_train_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  const fs::path db_file = dir / "train.tadah";
  {
    std::ofstream os(db_file);
    const StructureDB db = training_db();
    for (std::size_t s = 0; s < db.size(); ++s) db(s).dump_to_file(os);
  }
  const fs::path old_cwd = fs::current_path();
  fs::current_path(dir);
  Context tc = c;
  set_key(tc, "WDBFILE", "-1");
  set_key(tc, "DBFILE", db_file.string());
  std::string what = "no exception";
  {
    std::ostringstream captured;
    std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
    try {
      tadah::mlip::Task task("train", tc);
      tadah::mlip::engines::TrainEngine().run(task);
    } catch (const std::exception &e) {
      what = e.what();
    }
    std::cout.rdbuf(old_buf);
  }
  fs::current_path(old_cwd);
  std::error_code ec;
  fs::remove_all(dir, ec);
  CHECK_THAT(what, Contains("WDBFILE -1") && Contains(">= 0") &&
                       !Contains("intercept"));
}

/* Pins: a dataset file whose second structure has the weights line
 * '-1 1 1' is refused as it is read, naming the file, the structure (2),
 * its label and "eweight '-1' is negative"; '1 -0.5 1' names fweight.
 * Before the fix: both files were read, the negative weight kept. */
TEST_CASE("A dataset with a negative weight is refused as it is read",
          "[fit_weights][refusal]") {
  PeriodicTable::initialize();
  const fs::path dir = fs::temp_directory_path() /
                       ("tadah_fit_weights_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto write = [&](const std::string &name, const std::string &weights) {
    const fs::path p = dir / name;
    std::ofstream f(p);
    const char *cell = "4.0 0.0 0.0\n0.0 4.0 0.0\n0.0 0.0 4.0\n"
                       "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n";
    f << "first\n-8.0\n" << cell << "Ta 0.0 0.0 0.0 0.0 0.0 0.0\n\n";
    f << "second\n" << weights << "\n-8.1\n" << cell
      << "Ta 0.1 0.0 0.0 0.0 0.0 0.0\n";
    return p.string();
  };
  const std::string e = write("e.tadah", "-1 1 1");
  const std::string f = write("f.tadah", "1 -0.5 1");
  StructureDB db;
  CHECK_THAT(refusal([&] { db.add(e); }),
             Contains(e) && Contains("structure 2") && Contains("'second'") &&
                 Contains("eweight '-1' is negative"));
  CHECK_THAT(refusal([&] { db.add(f); }),
             Contains(f) && Contains("structure 2") &&
                 Contains("fweight '-0.5' is negative"));
  std::error_code ec;
  fs::remove_all(dir, ec);
}

/* Pins: a workspace manager asked for a design-matrix workspace with the
 * copy of the rows (wcopy true, what a fit keeps for predict(StructureDB&))
 * after one of the same size without it gives a workspace with the copy:
 * Phi_cpy and T_cpy of 4 x 2 and 4.
 * Before the fix: it gave the workspace without the copy (Phi_cpy 0 x 0),
 * which the build then wrote into. */
TEST_CASE("A design-matrix workspace without the copy of the rows is not "
          "reused for a build that needs it",
          "[fit_weights][workspace]") {
  MLIPWorkspaceManager m;
  m.getDesignMatrixWorkspace(4, 2, false);
  tadah::mlip::memory::DesignMatrixWorkspace *w = m.getDesignMatrixWorkspace(4, 2, true);
  CHECK(w->Phi_cpy.rows() == 4);
  CHECK(w->Phi_cpy.cols() == 2);
  CHECK(w->T_cpy.size() == 4);
}
