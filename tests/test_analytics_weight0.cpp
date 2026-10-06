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
// Synthetic data: dataset 1 holds A (weights 1) and B (weights 0, with
// reference forces and stress far from the prediction), dataset 2 holds C
// (weights 1). Every structure has 2 atoms, so 6 force components and 6
// stress components (xx, xy, xz, yy, yz, zz). The prediction differs from
// the reference by d:
//   A: one force component by 0.3, the stress xy by 0.01;
//   C: one force component by 0.4, the stress zz by 0.02;
//   B: every force component by 50, every stress component by 0.5.
// Before the fix B counted, so every metric of dataset 1 was off by O(10)
// (forces) or O(0.1) (stress), and an all-weight-0 dataset gave a number.

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

/* The reference: dataset 1 = {A, B}, dataset 2 = {C}. */
StructureDB reference() {
  StructureDB db;
  db.dbidx = {0, 2, 3};
  db.structures.resize(3);
  const char *labels[3] = {"A", "B", "C"};
  for (std::size_t i = 0; i < 3; ++i) {
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
  // B: no data of its own (weights 0), with values far from any prediction.
  db(1).fweight = 0.0;
  db(1).sweight = 0.0;
  for (std::size_t a = 0; a < 2; ++a) db(1).atoms[a].force = Vec3d(50.0, 50.0, 50.0);
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) db(1).stress(x, y) = 0.5;
  db(2).atoms[0].force = Vec3d(0.0, 0.0, 0.4);
  db(2).atoms[1].force = Vec3d(0.0, 0.0, 0.0);
  db(2).stress(2, 2) = 0.1;
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
  // B: d = 50 on every force component, 0.5 on every stress component.
  p(2).atoms[0].force = Vec3d(0.0, 0.0, 0.4);
  p(2).atoms[1].force = Vec3d(0.0, -0.4, 0.0);  // d = 0.4
  p(2).stress(2, 2) = 0.08;                     // d = 0.02
  return p;
}

/* R^2 = 1 - SS_res / SS_tot over the six components @p obs with the
 * residual sum of squares @p ss_res. */
double r_sq(const double (&obs)[6], double ss_res) {
  double mean = 0.0;
  for (double o : obs) mean += o / 6.0;
  double ss_tot = 0.0;
  for (double o : obs) ss_tot += (o - mean) * (o - mean);
  return 1.0 - ss_res / ss_tot;
}

}  // namespace

/* Pins: the force metrics of dataset 1 count A only (B has fweight 0):
 * MAE 0.3/6, RMSE sqrt(0.09/6), R^2 over A's six components; dataset 2:
 * MAE 0.4/6, RMSE sqrt(0.16/6).
 * Before the fix: B's twelve components of 50 were included. */
TEST_CASE("Analytics: the force metrics leave out structures with force weight 0",
          "[analytics][weight0]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  const auto mae = a.calc_f_mae();
  const auto rmse = a.calc_f_rmse();
  const auto rsq = a.calc_f_r_sq();
  REQUIRE(mae.size() == 2);
  CHECK(mae(0) == Approx(0.3 / 6).epsilon(kRel));
  CHECK(mae(1) == Approx(0.4 / 6).epsilon(kRel));
  CHECK(rmse(0) == Approx(std::sqrt(0.09 / 6)).epsilon(kRel));
  CHECK(rmse(1) == Approx(std::sqrt(0.16 / 6)).epsilon(kRel));
  const double obs_a[6] = {0.1, 0.0, 0.0, 0.0, 0.2, 0.0};
  CHECK(rsq(0) == Approx(r_sq(obs_a, 0.09)).epsilon(kRel));
}

/* Pins: the stress metrics of dataset 1 count A only (B has sweight 0):
 * MAE 0.01/6, RMSE sqrt(0.0001/6), R^2 over A's six components; dataset 2:
 * MAE 0.02/6, RMSE sqrt(0.0004/6).
 * Before the fix: B's six components of 0.5 were included. */
TEST_CASE("Analytics: the stress metrics leave out structures with stress weight 0",
          "[analytics][weight0]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  const auto mae = a.calc_s_mae();
  const auto rmse = a.calc_s_rmse();
  const auto rsq = a.calc_s_r_sq();
  REQUIRE(mae.size() == 2);
  CHECK(mae(0) == Approx(0.01 / 6).epsilon(kRel));
  CHECK(mae(1) == Approx(0.02 / 6).epsilon(kRel));
  CHECK(rmse(0) == Approx(std::sqrt(0.0001 / 6)).epsilon(kRel));
  CHECK(rmse(1) == Approx(std::sqrt(0.0004 / 6)).epsilon(kRel));
  const double obs_a[6] = {0.1, 0.0, 0.0, 0.2, 0.0, 0.3};
  CHECK(rsq(0) == Approx(r_sq(obs_a, 0.0001)).epsilon(kRel));
}

/* Pins: the PC_FRMSE partials (analytics_grad::frmse_per_dataset) and the
 * PC_SRMSE partials (analytics_grad::srmse) leave out B as the metrics do:
 * none is emitted for B, and A's are -d / (N metric) with N = 6 and the
 * dataset-1 metric over A (frmse_per_dataset does not average over the
 * datasets; srmse divides by their number, 2).
 * Before the fix: B's cells were emitted, and A's used N = 12 and a metric
 * that included B. */
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
  const double frmse1 = std::sqrt(0.09 / 6);
  REQUIRE(f.count({0, 0}) == 1);
  CHECK(f.at({0, 0}) == Approx(-0.3 / (6 * frmse1)).epsilon(kRel));
  CHECK(f.at({0, 4}) == Approx(0.0).margin(1e-15));

  std::map<std::pair<std::size_t, std::size_t>, double> s;
  analytics_grad::srmse(
      truth, pred, 1.0,
      [&](std::size_t i, PredCellKind kind, std::size_t comp, double v) {
        CHECK(kind == PredCellKind::STRESS);
        s[{i, comp}] = v;
      });
  for (const auto &cell : s) CHECK(cell.first.first != 1);
  const double srmse1 = std::sqrt(0.0001 / 6);
  REQUIRE(s.count({0, 1}) == 1);  // xy
  CHECK(s.at({0, 1}) == Approx(-0.01 / (2 * 6 * srmse1)).epsilon(kRel));
}

/* Pins: a dataset in which every structure has force weight 0 (stress
 * weight 0) has no force (stress) metric: each force (stress) metric and
 * partial is refused with a std::runtime_error naming the dataset ("dataset
 * 2 of 2") and the weight, also in calc_f_mae etc. With C's fweight 0 the
 * stress metrics are still measured, and the reverse.
 * Before the fix: they returned a number (B's or C's zero data compared
 * with the prediction). */
TEST_CASE("Analytics: a dataset whose structures all have weight 0 has no metric",
          "[analytics][weight0]") {
  const StructureDB pred = prediction();
  SECTION("force weight 0") {
    StructureDB truth = reference();
    truth(2).fweight = 0.0;
    const Analytics a(truth, pred);
    const auto names = Contains("dataset 2 of 2") && Contains("force weight 0");
    CHECK_THROWS_WITH(a.calc_f_mae(), names);
    CHECK_THROWS_WITH(a.calc_f_rmse(), names);
    CHECK_THROWS_WITH(a.calc_f_r_sq(), names);
    CHECK_THROWS_WITH(analytics_grad::frmse_per_dataset(
                          truth, pred, 1.0,
                          [](std::size_t, std::size_t, PredCellKind, std::size_t, double) {}),
                      names);
    CHECK(a.calc_s_rmse()(1) == Approx(std::sqrt(0.0004 / 6)).epsilon(kRel));
  }
  SECTION("stress weight 0") {
    StructureDB truth = reference();
    truth(2).sweight = 0.0;
    const Analytics a(truth, pred);
    const auto names = Contains("dataset 2 of 2") && Contains("stress weight 0");
    CHECK_THROWS_WITH(a.calc_s_mae(), names);
    CHECK_THROWS_WITH(a.calc_s_rmse(), names);
    CHECK_THROWS_WITH(a.calc_s_r_sq(), names);
    CHECK_THROWS_WITH(analytics_grad::srmse(
                          truth, pred, 1.0,
                          [](std::size_t, PredCellKind, std::size_t, double) {}),
                      names);
    CHECK(a.calc_f_rmse()(1) == Approx(std::sqrt(0.16 / 6)).epsilon(kRel));
  }
}
