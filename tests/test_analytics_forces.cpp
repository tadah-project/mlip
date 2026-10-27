// Force metrics need a predicted force for every reference atom.
//
// Analytics::calc_f_mae, calc_f_rmse and calc_f_r_sq, and
// analytics_grad::frmse_per_dataset (with its averaging wrapper
// analytics_grad::frmse), compare the reference force
// truth(i).atoms[a].force with the predicted pred(i).atoms[a].force for
// every reference atom a < truth(i).natoms(). A prediction made without
// forces -- M_Tadah_Base::predict with FORCE false never sizes
// Structure::atoms -- or with another number of atoms than the reference
// must be refused with a std::runtime_error whose message names the
// structure ("structure <k>", k the 1-based index in the StructureDB), the
// reference structure's label when it has one, and "forces". Before the
// fix (require_predicted_forces) these functions read past the end of the
// predicted atoms: undefined behaviour, a SIGSEGV for an empty vector (tadah
// hpo with PC_FRMSE and FORCE false) and a wrong number otherwise.
//
// Synthetic data, one dataset of two 2-atom structures. Every case starts
// with a complete prediction whose metric equals a hand-computed value (it
// passed before the fix too, and keeps the check from refusing good
// predictions); the refusal sections that follow failed before the fix.

#include "catch2/catch.hpp"

#include <tadah/core/vec3d.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/analytics_grad.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <cctype>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using tadah::core::Vec3d;
using tadah::mlip::Analytics;
using tadah::mlip::PredCellKind;
using tadah::mlip::StructureDB;
namespace analytics_grad = tadah::mlip::analytics_grad;

namespace {

const std::string kLabel1 = "rattled bcc Ta, cell A";
const std::string kLabel2 = "rattled bcc Ta, cell B";

/* Relative tolerance of the hand-computed metrics: each is a sum of 12
 * terms of order 0.1 in double precision (a few ulp, ~1e-15 relative);
 * a wrong formula moves the value by O(1). */
constexpr double kRel = 1e-12;

/* Reference forces [structure][atom]. The 12 components sum to exactly
 * zero (they cancel in pairs), so their mean is 0 and
 *   SS_tot = sum of squares = 2 (0.04 + 0.16 + 0.36 + 0.64 + 0.04) = 2.48. */
const Vec3d kRefForce[2][2] = {
    {Vec3d(0.2, -0.2, 0.4), Vec3d(-0.4, 0.6, -0.6)},
    {Vec3d(0.8, -0.8, 0.0), Vec3d(0.0, 0.2, -0.2)}};

/* Reference minus prediction, d = F_ref - F_pred [structure][atom]: three
 * components of magnitude 0.4, the other nine zero. Hence, over the N = 12
 * force components of the one dataset,
 *   MAE  = (3 * 0.4) / 12            = 0.1
 *   RMSE = sqrt(3 * 0.16 / 12)       = sqrt(0.04) = 0.2
 *   R^2  = 1 - SS_res / SS_tot = 1 - 0.48 / 2.48 = 25/31. */
const Vec3d kDiff[2][2] = {
    {Vec3d(0.4, 0.0, 0.0), Vec3d(0.0, 0.0, -0.4)},
    {Vec3d(0.0, 0.0, 0.0), Vec3d(0.0, 0.4, 0.0)}};

constexpr double kMae = 0.1;
constexpr double kRmse = 0.2;
constexpr double kRsq = 25.0 / 31.0;

/* The reference: one dataset of two labelled 2-atom structures. */
StructureDB reference() {
  StructureDB db;
  db.dbidx = {0, 2};
  db.structures.resize(2);
  for (std::size_t i = 0; i < 2; ++i) {
    db(i).label = i == 0 ? kLabel1 : kLabel2;
    db(i).energy = -1.5 * static_cast<double>(i + 1);
    db(i).atoms.resize(2);
    for (std::size_t a = 0; a < 2; ++a) db(i).atoms[a].force = kRefForce[i][a];
  }
  return db;
}

/* A prediction as M_Tadah_Base::predict makes it: energy set, no label,
 * the dataset layout of the reference, and no atoms (FORCE false). */
StructureDB prediction_without_forces() {
  const StructureDB ref = reference();
  StructureDB p;
  p.dbidx = ref.dbidx;
  p.structures.resize(ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) p(i).energy = ref(i).energy;
  return p;
}

/* The same prediction with forces (FORCE true): F_ref - kDiff. */
StructureDB prediction_with_forces() {
  StructureDB p = prediction_without_forces();
  for (std::size_t i = 0; i < 2; ++i) {
    p(i).atoms.resize(2);
    for (std::size_t a = 0; a < 2; ++a)
      p(i).atoms[a].force = kRefForce[i][a] - kDiff[i][a];
  }
  return p;
}

/* Whether @p msg names structure @p k: "structure <k>" not followed by
 * another digit (so structure 1 is not found in "structure 12"). */
bool names_structure(const std::string &msg, std::size_t k) {
  const std::string needle = "structure " + std::to_string(k);
  for (auto pos = msg.find(needle); pos != std::string::npos;
       pos = msg.find(needle, pos + 1)) {
    const auto end = pos + needle.size();
    if (end == msg.size() ||
        !std::isdigit(static_cast<unsigned char>(msg[end])))
      return true;
  }
  return false;
}

/* The message of the std::runtime_error that @p metric(truth, pred) throws;
 * fails the case when it returns. */
template <class Metric>
std::string refusal_of(Metric &&metric, const StructureDB &truth,
                       const StructureDB &pred) {
  try {
    metric(truth, pred);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  FAIL("the force metric accepted a prediction whose atoms do not match "
       "the reference");
  return {};
}

void check_names(const std::string &msg, std::size_t k,
                 const std::string &label, const std::string &head) {
  INFO("message: " << msg);
  CHECK(msg.rfind(head, 0) == 0);
  CHECK(names_structure(msg, k));
  if (!label.empty()) CHECK_THAT(msg, Catch::Matchers::Contains(label));
  CHECK_THAT(msg, Catch::Matchers::Contains("forces"));
}

/* The refusal sections every force metric shares. They run after the
 * metric's positive control. Order: first the cases that read past the
 * end without a crash before the fix, then those that dereferenced an
 * empty vector (a SIGSEGV before the fix, which ended the process). */
template <class Metric>
void refusal_sections(Metric metric, const std::string &head) {
  const StructureDB truth = reference();

  SECTION("structure 2 with more predicted atoms than the reference is "
          "refused") {
    StructureDB pred = prediction_with_forces();
    pred(1).atoms.resize(3);
    pred(1).atoms[2].force = Vec3d(0.0, 0.0, 0.0);
    check_names(refusal_of(metric, truth, pred), 2, kLabel2, head);
  }
  SECTION("structure 2 with fewer predicted atoms than the reference is "
          "refused") {
    StructureDB pred = prediction_with_forces();
    pred(1).atoms.resize(1);
    check_names(refusal_of(metric, truth, pred), 2, kLabel2, head);
  }
  SECTION("an unlabelled structure without forces is refused by its index") {
    StructureDB unlabelled = truth;
    unlabelled(1).label.clear();
    StructureDB pred = prediction_with_forces();
    std::vector<tadah::mlip::Atom>().swap(pred(1).atoms);
    check_names(refusal_of(metric, unlabelled, pred), 2, "", head);
  }
  SECTION("a prediction without forces is refused at structure 1") {
    const StructureDB pred = prediction_without_forces();
    check_names(refusal_of(metric, truth, pred), 1, kLabel1, head);
  }
}

/* The frmse partials, keyed by (structure, component): with scale 1,
 * N = 12 and RMSE 0.2, d(RMSE)/d(F_pred) = -d / (N * RMSE) = -d / 2.4,
 * so -1/6 or +1/6 at the three non-zero differences and 0 elsewhere. */
std::map<std::pair<std::size_t, std::size_t>, double> expected_partials() {
  std::map<std::pair<std::size_t, std::size_t>, double> out;
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t a = 0; a < 2; ++a)
      for (std::size_t k = 0; k < 3; ++k)
        out[{i, 3 * a + k}] = -kDiff[i][a](k) / 2.4;
  return out;
}

void check_partials(
    const std::map<std::pair<std::size_t, std::size_t>, double> &got,
    std::size_t calls) {
  const auto want = expected_partials();
  CHECK(calls == want.size());  // one call per cell
  REQUIRE(got.size() == want.size());
  for (const auto &[cell, value] : want) {
    INFO("structure " << cell.first << ", component " << cell.second);
    REQUIRE(got.count(cell) == 1);
    /* The zero cells are computed as exactly -0.0 (d = x - x); the margin
     * only lets Approx compare them. */
    CHECK(got.at(cell) == Approx(value).epsilon(kRel).margin(1e-15));
  }
}

} // namespace

// Pins: calc_f_mae refuses a prediction whose structure has no forces or
// another atom count, naming the metric ("the force MAE: "), the 1-based
// structure index, the reference label and "forces"; a complete
// prediction gives MAE 0.1 (calc_f_rmse: "the force RMSE: ", calc_f_r_sq:
// "the force R^2: ").
// Before the fix that names the metric: the messages began
// "Analytics::calc_f_mae: ", the function: the prefix checks fail.
// Before the fix the refusal sections failed (no throw, or a SIGSEGV on
// the empty atoms of the last two sections); removing the
// require_predicted_forces call from calc_f_mae makes them fail again.
TEST_CASE("Analytics::calc_f_mae refuses a prediction without matching "
          "forces",
          "[analytics][analytics_forces]") {
  SECTION("a prediction with forces gives the hand-computed MAE") {
    const StructureDB truth = reference();
    const StructureDB pred = prediction_with_forces();
    const auto v = Analytics(truth, pred).calc_f_mae();
    REQUIRE(v.size() == 1);
    CHECK(v[0] == Approx(kMae).epsilon(kRel));
  }
  refusal_sections(
      [](const StructureDB &t, const StructureDB &p) {
        (void)Analytics(t, p).calc_f_mae();
      },
      "the force MAE: ");
}

// Pins: as calc_f_mae, for calc_f_rmse (what HPO's fill_validation calls
// for PC_FRMSE); a complete prediction gives RMSE 0.2.
// Before the fix the refusal sections failed (no throw, or a SIGSEGV);
// removing the require_predicted_forces call from calc_f_rmse makes them
// fail again.
TEST_CASE("Analytics::calc_f_rmse refuses a prediction without matching "
          "forces",
          "[analytics][analytics_forces]") {
  SECTION("a prediction with forces gives the hand-computed RMSE") {
    const StructureDB truth = reference();
    const StructureDB pred = prediction_with_forces();
    const auto v = Analytics(truth, pred).calc_f_rmse();
    REQUIRE(v.size() == 1);
    CHECK(v[0] == Approx(kRmse).epsilon(kRel));
  }
  refusal_sections(
      [](const StructureDB &t, const StructureDB &p) {
        (void)Analytics(t, p).calc_f_rmse();
      },
      "the force RMSE: ");
}

// Pins: as calc_f_mae, for calc_f_r_sq; a complete prediction gives
// R^2 = 1 - 0.48/2.48 = 25/31.
// Before the fix the refusal sections failed (no throw, or a SIGSEGV);
// removing the require_predicted_forces call from calc_f_r_sq makes them
// fail again.
TEST_CASE("Analytics::calc_f_r_sq refuses a prediction without matching "
          "forces",
          "[analytics][analytics_forces]") {
  SECTION("a prediction with forces gives the hand-computed R^2") {
    const StructureDB truth = reference();
    const StructureDB pred = prediction_with_forces();
    const auto v = Analytics(truth, pred).calc_f_r_sq();
    REQUIRE(v.size() == 1);
    CHECK(v[0] == Approx(kRsq).epsilon(kRel));
  }
  refusal_sections(
      [](const StructureDB &t, const StructureDB &p) {
        (void)Analytics(t, p).calc_f_r_sq();
      },
      "the force R^2: ");
}

// Pins: as calc_f_mae, for analytics_grad::frmse_per_dataset (the
// PC_FRMSE gradient of GRAD_STRATEGY ANALYTICAL/HYBRID); a complete
// prediction gives the partials -d / (N * RMSE) = -d / 2.4 in dataset 0.
// Before the fix the refusal sections failed (no throw, or a SIGSEGV);
// removing the require_predicted_forces call from frmse_per_dataset makes
// them fail again.
// The refusal names the metric, "the gradient of the force RMSE: structure
// ...". Before the fix that names the metric: it began "analytics_grad::frmse:
// ", the function: the prefix checks fail.
TEST_CASE("analytics_grad::frmse_per_dataset refuses a prediction without "
          "matching forces",
          "[analytics][analytics_forces][pc_dataset]") {
  SECTION("a prediction with forces gives the hand-computed partials") {
    const StructureDB truth = reference();
    const StructureDB pred = prediction_with_forces();
    std::map<std::pair<std::size_t, std::size_t>, double> got;
    std::size_t calls = 0;
    analytics_grad::frmse_per_dataset(
        truth, pred, 1.0,
        [&](std::size_t dataset, std::size_t i, PredCellKind kind,
            std::size_t comp, double partial) {
          ++calls;
          CHECK(dataset == 0);
          CHECK(kind == PredCellKind::FORCE);
          got[{i, comp}] = partial;
        });
    check_partials(got, calls);
  }
  refusal_sections(
      [](const StructureDB &t, const StructureDB &p) {
        analytics_grad::frmse_per_dataset(
            t, p, 1.0,
            [](std::size_t, std::size_t, PredCellKind, std::size_t, double) {});
      },
      "the gradient of the force RMSE: ");
}

// Pins: the averaging wrapper analytics_grad::frmse refuses the same
// predictions (it must not bypass the check of frmse_per_dataset); with
// one dataset its partials equal those of frmse_per_dataset.
// Before the fix the refusal sections failed (no throw, or a SIGSEGV);
// removing the require_predicted_forces call from frmse_per_dataset, or a
// frmse that bypasses frmse_per_dataset, makes them fail again.
// The refusal names the metric, "the gradient of the force RMSE: structure
// ...". Before the fix that names the metric: it began "analytics_grad::frmse:
// ", the function: the prefix checks fail.
TEST_CASE("analytics_grad::frmse refuses a prediction without matching "
          "forces",
          "[analytics][analytics_forces][pc_dataset]") {
  SECTION("a prediction with forces gives the hand-computed partials") {
    const StructureDB truth = reference();
    const StructureDB pred = prediction_with_forces();
    std::map<std::pair<std::size_t, std::size_t>, double> got;
    std::size_t calls = 0;
    analytics_grad::frmse(truth, pred, 1.0,
                          [&](std::size_t i, PredCellKind kind,
                              std::size_t comp, double partial) {
                            ++calls;
                            CHECK(kind == PredCellKind::FORCE);
                            got[{i, comp}] = partial;
                          });
    check_partials(got, calls);
  }
  refusal_sections(
      [](const StructureDB &t, const StructureDB &p) {
        analytics_grad::frmse(
            t, p, 1.0, [](std::size_t, PredCellKind, std::size_t, double) {});
      },
      "the gradient of the force RMSE: ");
}
