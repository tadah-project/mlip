// test_analytics_energy_weight0.cpp
//
// The energy metrics leave out the structures stored with energy priority 0.
//
// A structure with eweight 0 has no energy of its own (its energy line,
// which the dataset format needs, holds a placeholder). Training leaves its
// energy out through the weight; the energy metrics (Analytics::calc_e_mae,
// calc_e_rmse, calc_e_rrmse, calc_e_r_sq: PC_EMAE, PC_ERMSE, PC_ErRMSE,
// PC_ERSQ of tadah hpo, the Energy lines of tadah predict -A) and their
// partials (analytics_grad::emae, ermse_per_dataset, ermse, errmse, ersq)
// leave it out too, as the force and stress metrics leave out the
// structures with fweight or sweight 0. A dataset in which every structure
// has eweight 0 is not measured: its metric is nan and the mean runs over
// the other datasets (MetricQuantity::Energy of measured_datasets,
// measured_mean, require_metric_structures).
//
// Synthetic data, every structure of 2 atoms. Dataset 1 holds A (energy off
// by 0.1 eV), Z (eweight 0, a placeholder energy off by 50 eV) and C (off by
// 0.3 eV); dataset 2 holds D (off by 0.2 eV), Y (eweight 0, off by 40 eV)
// and E (off by 0.4 eV); dataset 3 holds X and W (eweight 0, off by 30 and
// 20 eV). The reference energies are -4, 2, -3, -5, 1, -6, 3 and 1.5 eV (no
// placeholder is 0, which the relative RMSE leaves out by itself).

#include "catch2/catch.hpp"

#include <tadah/core/vec3d.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/analytics_grad.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::Contains;
using tadah::mlip::Analytics;
using tadah::mlip::MetricQuantity;
using tadah::mlip::PredCellKind;
using tadah::mlip::StructureDB;
namespace analytics_grad = tadah::mlip::analytics_grad;

namespace {

constexpr double kRel = 1e-12;

bool is_nan_bits(double v) {
  std::uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  return (b & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
         (b & UINT64_C(0x000fffffffffffff)) != 0;
}

/* The reference: dataset 1 = {A, Z, C}, dataset 2 = {D, Y, E}, dataset 3
 * = {X, W}; Z, Y, X and W have eweight 0. */
StructureDB reference() {
  StructureDB db;
  db.dbidx = {0, 3, 6, 8};
  const char *labels[8] = {"A", "Z", "C", "D", "Y", "E", "X", "W"};
  const double energy[8] = {-4.0, 2.0, -3.0, -5.0, 1.0, -6.0, 3.0, 1.5};
  const double eweight[8] = {1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0};
  db.structures.resize(8);
  for (std::size_t i = 0; i < 8; ++i) {
    db(i).label = labels[i];
    db(i).energy = energy[i];
    db(i).eweight = eweight[i];
    db(i).atoms.resize(2);
    db(i).stress.set_zero();
  }
  return db;
}

/* The prediction: the reference energy minus d. */
StructureDB prediction() {
  const double d[8] = {0.1, 50.0, 0.3, 0.2, 40.0, 0.4, 30.0, 20.0};
  StructureDB p = reference();
  for (std::size_t i = 0; i < 8; ++i) {
    p(i).energy -= d[i];
    p(i).eweight = 1.0;  // the prediction's weights are not read
  }
  return p;
}

/* The mean of the dataset values that are not nan. */
double mean_of_measured(const tadah::core::t_type &v) {
  double s = 0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    if (!is_nan_bits(v[i])) {
      s += v[i];
      ++n;
    }
  return s / static_cast<double>(n);
}

using Metric = std::function<tadah::core::t_type(const Analytics &)>;
using Grad = std::function<void(
    const StructureDB &, const StructureDB &, double,
    std::function<void(std::size_t, PredCellKind, std::size_t, double)>)>;

struct EnergyMetric {
  const char *name;
  Metric value;
  Grad grad;
};

std::vector<EnergyMetric> energy_metrics() {
  return {
      {"emae", [](const Analytics &a) { return a.calc_e_mae(); },
       [](const StructureDB &t, const StructureDB &p, double s, auto f) {
         analytics_grad::emae(t, p, s, f);
       }},
      {"ermse", [](const Analytics &a) { return a.calc_e_rmse(); },
       [](const StructureDB &t, const StructureDB &p, double s, auto f) {
         analytics_grad::ermse(t, p, s, f);
       }},
      {"errmse", [](const Analytics &a) { return a.calc_e_rrmse(); },
       [](const StructureDB &t, const StructureDB &p, double s, auto f) {
         analytics_grad::errmse(t, p, s, f);
       }},
      {"ersq", [](const Analytics &a) { return a.calc_e_r_sq(); },
       [](const StructureDB &t, const StructureDB &p, double s, auto f) {
         analytics_grad::ersq(t, p, s, f);
       }}};
}

}  // namespace

/* Pins: the energy metrics of dataset 1 count A and C only: MAE (0.05 +
 * 0.15) / 2 eV/atom, RMSE sqrt((0.05^2 + 0.15^2) / 2), relative RMSE
 * sqrt(((0.1/4)^2 + (0.3/3)^2) / 2), R^2 over A and C; dataset 2's count D
 * and E only; dataset 3, whose structures all have eweight 0, is not
 * measured: nan for every energy metric.
 * Before the fix: Z, Y, X and W were counted (MAE of dataset 1 (0.05 + 25
 * + 0.15) / 3), dataset 3 had values. */
TEST_CASE("Analytics: the energy metrics leave out structures with energy "
          "priority 0",
          "[analytics][weight0][energy]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  const auto mae = a.calc_e_mae();
  const auto rmse = a.calc_e_rmse();
  const auto rrmse = a.calc_e_rrmse();
  const auto rsq = a.calc_e_r_sq();
  REQUIRE(mae.size() == 3);
  CHECK(mae(0) == Approx((0.05 + 0.15) / 2).epsilon(kRel));
  CHECK(mae(1) == Approx(0.15).epsilon(kRel));
  CHECK(rmse(0) == Approx(std::sqrt((0.05 * 0.05 + 0.15 * 0.15) / 2)).epsilon(kRel));
  CHECK(rmse(1) == Approx(std::sqrt((0.01 + 0.04) / 2)).epsilon(kRel));
  CHECK(rrmse(0) ==
        Approx(std::sqrt((0.025 * 0.025 + 0.1 * 0.1) / 2)).epsilon(kRel));
  CHECK(rrmse(1) ==
        Approx(std::sqrt((0.04 * 0.04 + (0.4 / 6) * (0.4 / 6)) / 2)).epsilon(kRel));
  // R^2 over A and C: obs -2, -1.5; pred -2.05, -1.65.
  const double ss_res = 0.05 * 0.05 + 0.15 * 0.15;
  const double ss_tot = 2 * 0.25 * 0.25;
  CHECK(rsq(0) == Approx(1.0 - ss_res / ss_tot).epsilon(kRel));
  for (const auto &v : {mae, rmse, rrmse, rsq}) CHECK(is_nan_bits(v(2)));
}

/* Pins: measured_datasets(Energy) is {true, true, false};
 * measured_mean of the RMSE over it is the mean of datasets 1 and 2;
 * require_metric_structures(Energy) passes with them and refuses data in
 * which every structure has eweight 0, naming the files and 'energy
 * priority 0 (eweight)' ('energy weight 0 (eweight)' before the priorities
 * were named so).
 * Before the fix: MetricQuantity has no Energy (does not compile). */
TEST_CASE("The energy metrics measure the datasets with an energy",
          "[analytics][weight0][energy]") {
  StructureDB truth = reference();
  const Analytics a(truth, prediction());
  const std::vector<bool> m = tadah::mlip::measured_datasets(truth, MetricQuantity::Energy);
  CHECK(m == std::vector<bool>{true, true, false});
  CHECK(tadah::mlip::counts_in_metric(truth(0), MetricQuantity::Energy));
  CHECK_FALSE(tadah::mlip::counts_in_metric(truth(1), MetricQuantity::Energy));
  const auto rmse = a.calc_e_rmse();
  CHECK(tadah::mlip::measured_mean(rmse, m) ==
        Approx((rmse(0) + rmse(1)) / 2).epsilon(kRel));
  CHECK_NOTHROW(tadah::mlip::require_metric_structures(truth, MetricQuantity::Energy, "t"));
  for (std::size_t i = 0; i < truth.size(); ++i) truth(i).eweight = 0.0;
  CHECK_THROWS_WITH(
      tadah::mlip::require_metric_structures(truth, MetricQuantity::Energy, "t",
                                             {"a.tadah", "b.tadah", "c.tadah"}),
      Contains("t: no dataset has a structure with an energy") &&
          Contains("a.tadah") && Contains("energy priority 0 (eweight)"));
}

/* Pins: the partials of every energy metric emit no cell of Z, Y, X or W, and
 * the averaging ones divide by the 2 measured datasets: the emae partial of
 * A is -1 / (2 x 2 x 2) (sign of the error, 2 datasets, 2 structures, 2
 * atoms); ermse_per_dataset's of A is -0.05 / (2 x rmse_1 x 2) and emits
 * nothing for dataset 3.
 * Before the fix: Z, Y, X and W had partials, the averages divided by 3
 * and the structures counted were 3 per dataset. */
TEST_CASE("analytics_grad: the energy partials leave out structures with "
          "energy priority 0",
          "[analytics][weight0][energy]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  for (const EnergyMetric &em : energy_metrics()) {
    INFO(em.name);
    std::map<std::size_t, double> cells;
    em.grad(truth, pred, 1.0,
            [&](std::size_t i, PredCellKind kind, std::size_t, double v) {
              CHECK(kind == PredCellKind::ENERGY);
              cells[i] = v;
            });
    for (std::size_t i : {1, 4, 6, 7}) CHECK(cells.count(i) == 0);
    for (std::size_t i : {0, 2, 3, 5}) CHECK(cells.count(i) == 1);
  }
  std::map<std::size_t, double> mae;
  analytics_grad::emae(truth, pred, 1.0,
                       [&](std::size_t i, PredCellKind, std::size_t, double v) { mae[i] = v; });
  CHECK(mae.at(0) == Approx(-1.0 / 8).epsilon(kRel));
  const Analytics a(truth, pred);
  const double rmse1 = a.calc_e_rmse()(0);
  std::size_t cells_3 = 0;
  std::map<std::size_t, double> per;
  analytics_grad::ermse_per_dataset(
      truth, pred, 1.0,
      [&](std::size_t d, std::size_t i, PredCellKind, std::size_t, double v) {
        cells_3 += d == 2;
        per[i] = v;
      });
  CHECK(cells_3 == 0);
  CHECK(per.at(0) == Approx(-0.05 / (2 * rmse1 * 2)).epsilon(kRel));
}

/* A guard: the partial of each energy metric's mean over the measured
 * datasets equals its central finite difference (h 1e-5 eV, 1e-5 relative:
 * the placeholders make the metrics large, and their rounding is divided
 * by h)
 * at every structure, those with eweight 0 included (partial and
 * difference 0). It passes before the fix as well (both sides then counted
 * every structure); it fails when only one side leaves a structure of
 * eweight 0 out (shown by removing the skip from ermse_per_dataset: the
 * partials of Z and Y were then not 0). */
TEST_CASE("analytics_grad: the energy partials equal finite differences with "
          "structures of energy priority 0",
          "[analytics][weight0][energy][guard]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const double h = 1e-5;
  for (const EnergyMetric &em : energy_metrics()) {
    INFO(em.name);
    std::map<std::size_t, double> cells;
    em.grad(truth, pred, 1.0,
            [&](std::size_t i, PredCellKind, std::size_t, double v) { cells[i] += v; });
    for (std::size_t i = 0; i < truth.size(); ++i) {
      INFO("structure " << i);
      StructureDB plus = pred, minus = pred;
      plus(i).energy += h;
      minus(i).energy -= h;
      const double fd = (mean_of_measured(em.value(Analytics(truth, plus))) -
                         mean_of_measured(em.value(Analytics(truth, minus)))) /
                        (2 * h);
      const double partial = cells.count(i) ? cells.at(i) : 0.0;
      CHECK(partial == Approx(fd).epsilon(1e-5).margin(1e-9));
    }
  }
}
