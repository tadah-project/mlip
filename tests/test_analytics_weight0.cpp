// The force and stress metrics leave out structures stored with weight 0.
//
// A structure with force weight 0 (fweight) has no force data of its own
// (tadah data convert stores it with zero forces, e.g. a .castep step that
// prints no forces), and one with stress weight 0 (sweight) no stress
// tensor. Training ignores them through the weights; the metrics must too
// (MK, 2026-10-05): Analytics::calc_f_mae/calc_f_rmse/calc_f_r_sq skip a
// structure with fweight 0, calc_s_mae/calc_s_rmse/calc_s_r_sq one with
// sweight 0, and so do the PC_FRMSE/PC_SRMSE partials of analytics_grad. A
// dataset in which every structure has that weight 0 is refused, naming the
// dataset: its metric would have nothing to measure.
//
// Synthetic data: dataset 1 holds A (weights 1), Bf (force weight 0, with
// reference forces far from the prediction and an exact stress) and Bs
// (stress weight 0, with a reference stress far from the prediction and
// exact forces); dataset 2 holds C (weights 1). So a metric that read the
// other weight would count Bf in the force metrics or Bs in the stress
// metrics. Every structure has 2 atoms, so 6 force components and 6 stress
// components (xx, xy, xz, yy, yz, zz). The prediction differs from the
// reference by d:
//   A:  one force component by 0.3, the stress xy by 0.01;
//   Bf: every force component by 50 (stress exact), its energy by 0.5;
//   Bs: every stress component by 0.5 (forces exact);
//   C:  one force component by 0.4, the stress zz by 0.02.
// The force metrics of dataset 1 count A and Bs (N = 12), its stress metrics
// A and Bf (N = 12); the energy metrics count all three.
// Before the fix Bf counted in the force metrics and Bs in the stress
// metrics, so the force MAE and RMSE of dataset 1 were off by O(10) and the
// stress ones by O(0.1), and an all-weight-0 dataset gave a number.

#include "catch2/catch.hpp"

#include <tadah/core/vec3d.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/analytics_grad.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

using Catch::Matchers::Contains;
using tadah::core::Vec3d;
using tadah::mlip::Analytics;
using tadah::mlip::PredCellKind;
using tadah::mlip::StructureDB;
namespace analytics_grad = tadah::mlip::analytics_grad;

namespace {

constexpr double kRel = 1e-12;

/* The reference: dataset 1 = {A, Bf, Bs}, dataset 2 = {C}. */
StructureDB reference() {
  StructureDB db;
  db.dbidx = {0, 3, 4};
  db.structures.resize(4);
  const char *labels[4] = {"A", "Bf", "Bs", "C"};
  for (std::size_t i = 0; i < 4; ++i) {
    db(i).label = labels[i];
    db(i).energy = -1.0;
    db(i).atoms.resize(2);
    db(i).stress.set_zero();
  }
  db(0).atoms[0].force = Vec3d(0.1, 0.0, 0.0);
  db(0).atoms[1].force = Vec3d(0.0, 0.2, 0.0);
  db(0).stress(0, 0) = 0.1;
  db(0).stress(1, 1) = 0.2;
  db(0).stress(2, 2) = 0.3;
  // Bf: no forces of its own (force weight 0), far from any prediction.
  db(1).fweight = 0.0;
  for (std::size_t a = 0; a < 2; ++a) db(1).atoms[a].force = Vec3d(50.0, 50.0, 50.0);
  db(1).stress(0, 0) = db(1).stress(1, 1) = db(1).stress(2, 2) = 0.2;
  // Bs: no stress tensor of its own (stress weight 0), far from any
  // prediction.
  db(2).sweight = 0.0;
  db(2).atoms[0].force = Vec3d(0.05, 0.0, 0.0);
  db(2).atoms[1].force = Vec3d(0.0, 0.0, -0.05);
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) db(2).stress(x, y) = 0.5;
  db(3).atoms[0].force = Vec3d(0.0, 0.0, 0.4);
  db(3).atoms[1].force = Vec3d(0.0, 0.0, 0.0);
  db(3).stress(2, 2) = 0.1;
  return db;
}

/* The prediction: reference minus d (see the file comment). */
StructureDB prediction() {
  const StructureDB ref = reference();
  StructureDB p;
  p.dbidx = ref.dbidx;
  p.structures.resize(ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    p(i).energy = ref(i).energy;
    p(i).atoms.resize(2);
    p(i).stress.set_zero();
  }
  p(0).atoms[0].force = Vec3d(-0.2, 0.0, 0.0);  // d = 0.3
  p(0).atoms[1].force = Vec3d(0.0, 0.2, 0.0);
  p(0).stress(0, 0) = 0.1;
  p(0).stress(0, 1) = p(0).stress(1, 0) = -0.01;  // d = 0.01
  p(0).stress(1, 1) = 0.2;
  p(0).stress(2, 2) = 0.3;
  // Bf: d = 50 on every force component (predicted 0), the stress exact,
  // the energy off by 0.5.
  p(1).energy = -1.5;
  p(1).stress(0, 0) = p(1).stress(1, 1) = p(1).stress(2, 2) = 0.2;
  // Bs: d = 0.5 on every stress component (predicted 0), the forces exact.
  p(2).atoms[0].force = Vec3d(0.05, 0.0, 0.0);
  p(2).atoms[1].force = Vec3d(0.0, 0.0, -0.05);
  p(3).atoms[0].force = Vec3d(0.0, 0.0, 0.4);
  p(3).atoms[1].force = Vec3d(0.0, -0.4, 0.0);  // d = 0.4
  p(3).stress(2, 2) = 0.08;                     // d = 0.02
  return p;
}

/* R^2 = 1 - SS_res / SS_tot over the twelve components @p obs with the
 * residual sum of squares @p ss_res. */
double r_sq(const double (&obs)[12], double ss_res) {
  double mean = 0.0;
  for (double o : obs) mean += o / 12.0;
  double ss_tot = 0.0;
  for (double o : obs) ss_tot += (o - mean) * (o - mean);
  return 1.0 - ss_res / ss_tot;
}

}  // namespace

/* Pins: the force metrics of dataset 1 count A and Bs (Bf has fweight 0):
 * MAE 0.3/12, RMSE sqrt(0.09/12), R^2 over their twelve components; dataset
 * 2: MAE 0.4/6, RMSE sqrt(0.16/6).
 * Before the fix: Bf's six components of 50 were included. */
TEST_CASE("Analytics: the force metrics leave out structures with force weight 0",
          "[analytics][weight0]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  const auto mae = a.calc_f_mae();
  const auto rmse = a.calc_f_rmse();
  const auto rsq = a.calc_f_r_sq();
  REQUIRE(mae.size() == 2);
  CHECK(mae(0) == Approx(0.3 / 12).epsilon(kRel));
  CHECK(mae(1) == Approx(0.4 / 6).epsilon(kRel));
  CHECK(rmse(0) == Approx(std::sqrt(0.09 / 12)).epsilon(kRel));
  CHECK(rmse(1) == Approx(std::sqrt(0.16 / 6)).epsilon(kRel));
  const double obs[12] = {0.1, 0.0, 0.0, 0.0, 0.2, 0.0,     // A
                          0.05, 0.0, 0.0, 0.0, 0.0, -0.05};  // Bs
  CHECK(rsq(0) == Approx(r_sq(obs, 0.09)).epsilon(kRel));
}

/* Pins: the stress metrics of dataset 1 count A and Bf (Bs has sweight 0):
 * MAE 0.01/12, RMSE sqrt(0.0001/12), R^2 over their twelve components;
 * dataset 2: MAE 0.02/6, RMSE sqrt(0.0004/6).
 * Before the fix: Bs's six components of 0.5 were included. */
TEST_CASE("Analytics: the stress metrics leave out structures with stress weight 0",
          "[analytics][weight0]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  const auto mae = a.calc_s_mae();
  const auto rmse = a.calc_s_rmse();
  const auto rsq = a.calc_s_r_sq();
  REQUIRE(mae.size() == 2);
  CHECK(mae(0) == Approx(0.01 / 12).epsilon(kRel));
  CHECK(mae(1) == Approx(0.02 / 6).epsilon(kRel));
  CHECK(rmse(0) == Approx(std::sqrt(0.0001 / 12)).epsilon(kRel));
  CHECK(rmse(1) == Approx(std::sqrt(0.0004 / 6)).epsilon(kRel));
  const double obs[12] = {0.1, 0.0, 0.0, 0.2, 0.0, 0.3,   // A
                          0.2, 0.0, 0.0, 0.2, 0.0, 0.2};  // Bf
  CHECK(rsq(0) == Approx(r_sq(obs, 0.0001)).epsilon(kRel));
}

/* Pins: the energy metrics count every structure, whatever its force and
 * stress weights: dataset 1's energy MAE is 0.25/3 eV/atom and RMSE
 * sqrt(0.0625/3) (Bf's energy is off by 0.5 eV, 0.25 eV/atom; A and Bs are
 * exact).
 * A pin: a weight-0 skip in the energy metrics fails it. */
TEST_CASE("Analytics: the energy metrics count structures with force or stress weight 0",
          "[analytics][weight0]") {
  const StructureDB truth = reference();  // Analytics keeps references
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  CHECK(a.calc_e_mae()(0) == Approx(0.25 / 3).epsilon(kRel));
  CHECK(a.calc_e_rmse()(0) == Approx(std::sqrt(0.0625 / 3)).epsilon(kRel));
}

/* Pins: the PC_FRMSE partials (analytics_grad::frmse_per_dataset) and the
 * PC_SRMSE partials (analytics_grad::srmse) leave out what the metrics leave
 * out: no force partial for Bf, no stress partial for Bs, and A's are
 * -d / (N metric) with N = 12 and the dataset-1 metric (frmse_per_dataset
 * does not average over the datasets; srmse divides by their number, 2).
 * Before the fix: Bf's force cells and Bs's stress cells were emitted, and
 * the metrics in the denominators included them. */
TEST_CASE("analytics_grad: the PC_FRMSE and PC_SRMSE partials leave out structures "
          "with weight 0",
          "[analytics][weight0][pc_dataset]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  std::map<std::pair<std::size_t, std::size_t>, double> f;
  analytics_grad::frmse_per_dataset(
      truth, pred, 1.0,
      [&](std::size_t, std::size_t i, PredCellKind kind, std::size_t comp, double v) {
        CHECK(kind == PredCellKind::FORCE);
        f[{i, comp}] = v;
      });
  for (const auto &cell : f) CHECK(cell.first.first != 1);
  CHECK(f.count({2, 0}) == 1);  // Bs counts in the force metrics
  const double frmse1 = std::sqrt(0.09 / 12);
  REQUIRE(f.count({0, 0}) == 1);
  CHECK(f.at({0, 0}) == Approx(-0.3 / (12 * frmse1)).epsilon(kRel));
  CHECK(f.at({0, 4}) == Approx(0.0).margin(1e-15));

  std::map<std::pair<std::size_t, std::size_t>, double> s;
  analytics_grad::srmse(
      truth, pred, 1.0,
      [&](std::size_t i, PredCellKind kind, std::size_t comp, double v) {
        CHECK(kind == PredCellKind::STRESS);
        s[{i, comp}] = v;
      });
  for (const auto &cell : s) CHECK(cell.first.first != 2);
  CHECK(s.count({1, 0}) == 1);  // Bf counts in the stress metrics
  const double srmse1 = std::sqrt(0.0001 / 12);
  REQUIRE(s.count({0, 1}) == 1);  // xy
  CHECK(s.at({0, 1}) == Approx(-0.01 / (2 * 12 * srmse1)).epsilon(kRel));
}

/* Pins: a dataset in which every structure has force weight 0 (stress
 * weight 0) has no force (stress) metric: each force (stress) metric and
 * partial is refused with a std::runtime_error naming the dataset ("dataset
 * 2 of 2") and the weight. With C's fweight 0 the stress metrics and
 * partial are still measured, and the reverse.
 * Before the fix: they returned a number (C's data compared with the
 * prediction, although its weight said it had none). */
TEST_CASE("Analytics: a dataset whose structures all have weight 0 has no metric",
          "[analytics][weight0]") {
  const StructureDB pred = prediction();
  SECTION("force weight 0") {
    StructureDB truth = reference();
    truth(3).fweight = 0.0;  // C
    const Analytics a(truth, pred);
    const auto names = Contains("dataset 2 of 2") && Contains("force weight 0");
    CHECK_THROWS_WITH(a.calc_f_mae(), names);
    CHECK_THROWS_WITH(a.calc_f_rmse(), names);
    CHECK_THROWS_WITH(a.calc_f_r_sq(), names);
    CHECK_THROWS_WITH(analytics_grad::frmse_per_dataset(
                          truth, pred, 1.0,
                          [](std::size_t, std::size_t, PredCellKind, std::size_t, double) {}),
                      names);
    const Analytics b(truth, pred);
    CHECK(b.calc_s_mae()(1) == Approx(0.02 / 6).epsilon(kRel));
    CHECK(b.calc_s_rmse()(1) == Approx(std::sqrt(0.0004 / 6)).epsilon(kRel));
    CHECK_NOTHROW(b.calc_s_r_sq());
    CHECK_NOTHROW(analytics_grad::srmse(
        truth, pred, 1.0, [](std::size_t, PredCellKind, std::size_t, double) {}));
  }
  SECTION("stress weight 0") {
    StructureDB truth = reference();
    truth(3).sweight = 0.0;  // C
    const Analytics a(truth, pred);
    const auto names = Contains("dataset 2 of 2") && Contains("stress weight 0");
    CHECK_THROWS_WITH(a.calc_s_mae(), names);
    CHECK_THROWS_WITH(a.calc_s_rmse(), names);
    CHECK_THROWS_WITH(a.calc_s_r_sq(), names);
    CHECK_THROWS_WITH(analytics_grad::srmse(
                          truth, pred, 1.0,
                          [](std::size_t, PredCellKind, std::size_t, double) {}),
                      names);
    const Analytics b(truth, pred);
    CHECK(b.calc_f_mae()(1) == Approx(0.4 / 6).epsilon(kRel));
    CHECK(b.calc_f_rmse()(1) == Approx(std::sqrt(0.16 / 6)).epsilon(kRel));
    CHECK_NOTHROW(b.calc_f_r_sq());
    CHECK_NOTHROW(analytics_grad::frmse_per_dataset(
        truth, pred, 1.0,
        [](std::size_t, std::size_t, PredCellKind, std::size_t, double) {}));
  }
}
