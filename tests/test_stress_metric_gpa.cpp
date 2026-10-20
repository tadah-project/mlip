// The stress metrics measure the stress tensor: its nine entries, in GPa.
//
// A residual stress tensor dS is measured by the root mean square of its
// nine entries, sqrt(sum_ab dS_ab^2 / 9), the Frobenius norm a fit
// minimises (row_weights.h: the off-diagonal rows xy, xz, yz with m = 2),
// so a rotated cell has the metric of the cell. Of a symmetric tensor the
// six entries x <= y are read, an off-diagonal one twice. The data hold
// stresses in eV/A^3; the metrics are in GPa (1 GPa = 6.241509074e-3
// eV/A^3, kGPaInEvPerA3):
//
//   Analytics::calc_s_rmse_gpa  sqrt(sum m dS^2 / (9 n)) / kGPaInEvPerA3
//   Analytics::calc_s_mae_gpa   sum m |dS| / (9 n) / kGPaInEvPerA3
//   Analytics::calc_s_r_sq      R^2 over the nine entries of each tensor
//
// over the n structures of a dataset that have a stress tensor (sweight not
// 0). analytics_grad::srmse_gpa_per_dataset gives the partial of each
// dataset's RMSE, analytics_grad::srmse_gpa that of their mean over the
// measured datasets, with respect to each of the six predicted entries
// x <= y (stress_cells_per_struct: an off-diagonal entry of a symmetric
// prediction is one value): -c^2 m dS / (9 n metric), c = 1/kGPaInEvPerA3.
// tadah predict -A prints these metrics; write_hpotarget's starter target
// weights the validation RMSEs by 1/a^2 for the accuracies a of the
// default error scales.
//
// The oracles below are written out from these formulas; the
// finite-difference case is a guard (it passes for any metric whose
// gradient is consistent) and was shown to catch a gradient that drops
// the factor m of the off-diagonal entries.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/vec3d.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/analytics_grad.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::Vec3d;
using tadah::mlip::Analytics;
using tadah::mlip::Atom;
using tadah::mlip::PredCellKind;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::kGPaInEvPerA3;
namespace analytics_grad = tadah::mlip::analytics_grad;
namespace fs = std::filesystem;

namespace {

constexpr double kRel = 1e-12;
constexpr double kC = 1.0 / kGPaInEvPerA3;  // eV/A^3 -> GPa

/* Whether @p v is a NaN, from its bits (-ffast-math may fold std::isnan). */
bool is_nan_bits(double v) {
  std::uint64_t b;
  std::memcpy(&b, &v, sizeof b);
  return (b & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
         (b & UINT64_C(0x000fffffffffffff)) != 0;
}

/* A symmetric stress tensor from its six entries x <= y. */
void set_stress(Structure &s, double xx, double xy, double xz, double yy,
                double yz, double zz) {
  s.stress(0, 0) = xx;
  s.stress(0, 1) = s.stress(1, 0) = xy;
  s.stress(0, 2) = s.stress(2, 0) = xz;
  s.stress(1, 1) = yy;
  s.stress(1, 2) = s.stress(2, 1) = yz;
  s.stress(2, 2) = zz;
}

/* Two datasets: {P, Q, Z} and {R}; Z has sweight 0 (its tensor is a
 * placeholder far from any prediction). One atom each. */
StructureDB reference() {
  StructureDB db;
  db.dbidx = {0, 3, 4};
  db.structures.resize(4);
  const char *labels[4] = {"P", "Q", "Z", "R"};
  for (std::size_t i = 0; i < 4; ++i) {
    db(i).label = labels[i];
    db(i).energy = -1.0;
    db(i).atoms.resize(1);
    db(i).stress.set_zero();
  }
  set_stress(db(0), 0.010, 0.002, -0.001, 0.020, 0.003, 0.030);
  set_stress(db(1), -0.004, 0.0, 0.005, 0.006, -0.002, 0.001);
  db(2).sweight = 0.0;
  set_stress(db(2), 9.0, 9.0, 9.0, 9.0, 9.0, 9.0);
  set_stress(db(3), 0.002, -0.003, 0.004, -0.005, 0.006, -0.007);
  return db;
}

/* The prediction: every entry of P and Q off by the residuals below. */
StructureDB prediction() {
  const StructureDB ref = reference();
  StructureDB p;
  p.dbidx = ref.dbidx;
  p.structures.resize(ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i) {
    p(i).energy = ref(i).energy;
    p(i).atoms.resize(1);
    p(i).stress.set_zero();
  }
  // P: dS (truth - pred) = (0.001, 0.002, -0.003, 0.0, 0.001, 0.002)
  set_stress(p(0), 0.009, 0.000, 0.002, 0.020, 0.002, 0.028);
  // Q: dS = (0.0, -0.001, 0.0, 0.002, 0.0, 0.0)
  set_stress(p(1), -0.004, 0.001, 0.005, 0.004, -0.002, 0.001);
  // Z: predicted 0 (its tensor counts nowhere)
  // R: dS = (0.0, 0.004, 0.0, 0.0, 0.0, -0.001)
  set_stress(p(3), 0.002, -0.007, 0.004, -0.005, 0.006, -0.006);
  return p;
}

/* The six residuals x <= y of P, Q and R (xx, xy, xz, yy, yz, zz). */
const double kP[6] = {0.001, 0.002, -0.003, 0.0, 0.001, 0.002};
const double kQ[6] = {0.0, -0.001, 0.0, 0.002, 0.0, 0.0};
const double kR[6] = {0.0, 0.004, 0.0, 0.0, 0.0, -0.001};
const double kM[6] = {1, 2, 2, 1, 2, 1};  // an off-diagonal entry twice

double sum_m_sq(const double (&d)[6]) {
  double s = 0.0;
  for (int k = 0; k < 6; ++k) s += kM[k] * d[k] * d[k];
  return s;
}
double sum_m_abs(const double (&d)[6]) {
  double s = 0.0;
  for (int k = 0; k < 6; ++k) s += kM[k] * std::fabs(d[k]);
  return s;
}

/* The nine entries of a symmetric tensor of six entries x <= y. */
std::vector<double> nine(const Structure &s) {
  std::vector<double> v;
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) v.push_back(s.stress(x, y));
  return v;
}

/* R^2 = 1 - SS_res / SS_tot over the entries @p obs, @p pred. */
double r_sq(const std::vector<double> &obs, const std::vector<double> &pred) {
  double mean = 0.0;
  for (double o : obs) mean += o / static_cast<double>(obs.size());
  double ss_tot = 0.0, ss_res = 0.0;
  for (std::size_t i = 0; i < obs.size(); ++i) {
    ss_tot += (obs[i] - mean) * (obs[i] - mean);
    ss_res += (obs[i] - pred[i]) * (obs[i] - pred[i]);
  }
  return 1.0 - ss_res / ss_tot;
}

struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_stress_gpa_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScratchCwd() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

std::string slurp(const fs::path &p) {
  std::ifstream in(p);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* What TaskFactory gets from the command line. */
struct CommandInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  CommandInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* The whitespace-separated fields of the line of @p out that starts with
 * @p metric ("Stress RMSE"), without the name's two words. */
std::vector<double> metric_values(const std::string &out,
                                  const std::string &metric) {
  std::istringstream is(out);
  for (std::string line; std::getline(is, line);) {
    if (line.rfind(metric, 0) != 0) continue;
    std::istringstream ls(line.substr(metric.size()));
    std::vector<double> v;
    for (double x; ls >> x;) v.push_back(x);
    return v;
  }
  return {};
}

}  // namespace

/* Pins: the RMSE of dataset 1 is sqrt(sum m dS^2 / 18) in GPa over P and Q
 * (Z has sweight 0), that of dataset 2 sqrt(sum m dS^2 / 9) over R; the MAE
 * sum m |dS| / (9 n) in GPa; R^2 over the nine entries of P and Q.
 * Before the fix: calc_s_rmse_gpa and calc_s_mae_gpa did not exist (the
 * stress metrics read six entries in eV/A^3), and calc_s_r_sq read six. */
TEST_CASE("Analytics: the stress metrics read the nine entries of the "
          "tensor, in GPa",
          "[analytics][stress_gpa]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  const auto rmse = a.calc_s_rmse_gpa();
  const auto mae = a.calc_s_mae_gpa();
  const auto rsq = a.calc_s_r_sq();
  REQUIRE(rmse.size() == 2);
  CHECK(rmse(0) ==
        Approx(kC * std::sqrt((sum_m_sq(kP) + sum_m_sq(kQ)) / 18.0)).epsilon(kRel));
  CHECK(rmse(1) == Approx(kC * std::sqrt(sum_m_sq(kR) / 9.0)).epsilon(kRel));
  CHECK(mae(0) ==
        Approx(kC * (sum_m_abs(kP) + sum_m_abs(kQ)) / 18.0).epsilon(kRel));
  CHECK(mae(1) == Approx(kC * sum_m_abs(kR) / 9.0).epsilon(kRel));
  std::vector<double> obs = nine(truth(0)), prd = nine(pred(0));
  for (double v : nine(truth(1))) obs.push_back(v);
  for (double v : nine(pred(1))) prd.push_back(v);
  CHECK(rsq(0) == Approx(r_sq(obs, prd)).epsilon(kRel));
  CHECK(rsq(1) == Approx(r_sq(nine(truth(3)), nine(pred(3)))).epsilon(kRel));
}

/* Pins: the RMSE of a cell and of the same cell rotated are one number:
 * the residual tensor of a rotation R dS R^T has the Frobenius norm of
 * dS (here a rotation by 30 degrees about z).
 * Before the fix: calc_s_rmse_gpa did not exist; the six-entry metric gives
 * the rotated cell another value. */
TEST_CASE("Analytics: the stress RMSE of a rotated cell is the cell's",
          "[analytics][stress_gpa]") {
  StructureDB truth, pred;
  truth.dbidx = pred.dbidx = {0, 1};
  truth.structures.resize(1);
  pred.structures.resize(1);
  truth(0).atoms.resize(1);
  pred(0).atoms.resize(1);
  truth(0).stress.set_zero();
  pred(0).stress.set_zero();
  set_stress(pred(0), 0.004, -0.001, 0.0, 0.002, 0.0, -0.003);  // dS = -pred
  const double rmse = Analytics(truth, pred).calc_s_rmse_gpa()(0);
  const double t = M_PI / 6.0, c = std::cos(t), s = std::sin(t);
  const double R[3][3] = {{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}};
  StructureDB rot = pred;
  for (int x = 0; x < 3; ++x)
    for (int y = 0; y < 3; ++y) {
      double v = 0.0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) v += R[x][i] * pred(0).stress(i, j) * R[y][j];
      rot(0).stress(x, y) = v;
    }
  CHECK(Analytics(truth, rot).calc_s_rmse_gpa()(0) == Approx(rmse).epsilon(1e-12));
}

/* Pins: a dataset of only structures of sweight 0 is not measured: nan. */
TEST_CASE("Analytics: the GPa stress metrics leave out a dataset without a "
          "stress tensor",
          "[analytics][stress_gpa]") {
  StructureDB truth = reference();
  truth(3).sweight = 0.0;
  const StructureDB pred = prediction();
  const Analytics a(truth, pred);
  CHECK(is_nan_bits(a.calc_s_rmse_gpa()(1)));
  CHECK(is_nan_bits(a.calc_s_mae_gpa()(1)));
  CHECK(is_nan_bits(a.calc_s_r_sq()(1)));
  CHECK(a.calc_s_rmse_gpa()(0) ==
        Approx(kC * std::sqrt((sum_m_sq(kP) + sum_m_sq(kQ)) / 18.0)).epsilon(kRel));
}

/* Pins: the partial of each dataset's RMSE with respect to a predicted
 * entry x <= y is -c^2 m dS / (9 n metric) (srmse_gpa_per_dataset, one per
 * dataset, nothing for Z), and srmse_gpa's that of the mean over the two
 * measured datasets, half of it; six cells per structure.
 * Before the fix: neither function existed (analytics_grad::srmse took six
 * entries in eV/A^3). */
TEST_CASE("analytics_grad: the partials of the GPa stress RMSE",
          "[analytics][stress_gpa]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  CHECK(analytics_grad::stress_cells_per_struct(truth(0)) == 6);
  const double m1 = kC * std::sqrt((sum_m_sq(kP) + sum_m_sq(kQ)) / 18.0);
  const double m2 = kC * std::sqrt(sum_m_sq(kR) / 9.0);
  std::map<std::pair<std::size_t, std::size_t>, double> per, mean;
  analytics_grad::srmse_gpa_per_dataset(
      truth, pred, 1.0,
      [&](std::size_t d, std::size_t i, PredCellKind kind, std::size_t comp,
          double v) {
        CHECK(kind == PredCellKind::STRESS);
        CHECK(d == (i == 3 ? 1u : 0u));
        per[{i, comp}] = v;
      });
  analytics_grad::srmse_gpa(
      truth, pred, 1.0,
      [&](std::size_t i, PredCellKind, std::size_t comp, double v) {
        mean[{i, comp}] = v;
      });
  for (const auto &cell : per) CHECK(cell.first.first != 2);
  CHECK(per.size() == 18);
  for (int k = 0; k < 6; ++k) {
    INFO("entry " << k);
    const double p = -kC * kC * kM[k] * kP[k] / (18.0 * m1);
    const double r = -kC * kC * kM[k] * kR[k] / (9.0 * m2);
    CHECK(per.at({0, std::size_t(k)}) == Approx(p).margin(1e-12));
    CHECK(per.at({3, std::size_t(k)}) == Approx(r).margin(1e-12));
    CHECK(mean.at({0, std::size_t(k)}) == Approx(p / 2.0).margin(1e-12));
    CHECK(mean.at({3, std::size_t(k)}) == Approx(r / 2.0).margin(1e-12));
  }
}

/* A guard: the partials of srmse_gpa equal central finite differences of
 * the mean of calc_s_rmse_gpa over the measured datasets, for every
 * predicted entry x <= y of every structure (the prediction of an
 * off-diagonal entry is one value, as srmse_gpa's cell). It passes for any
 * consistent pair; a gradient without the factor m of the off-diagonal
 * entries fails it. */
TEST_CASE("analytics_grad: the GPa stress RMSE partials match finite "
          "differences",
          "[analytics][stress_gpa]") {
  const StructureDB truth = reference();
  const StructureDB pred = prediction();
  auto mean_rmse = [&](const StructureDB &p) {
    const auto v = Analytics(truth, p).calc_s_rmse_gpa();
    return 0.5 * (v(0) + v(1));
  };
  std::map<std::pair<std::size_t, std::size_t>, double> g;
  analytics_grad::srmse_gpa(
      truth, pred, 1.0,
      [&](std::size_t i, PredCellKind, std::size_t comp, double v) {
        g[{i, comp}] = v;
      });
  const double h = 1e-7;
  for (std::size_t i : {0u, 1u, 3u}) {
    std::size_t comp = 0;
    for (int x = 0; x < 3; ++x)
      for (int y = x; y < 3; ++y, ++comp) {
        StructureDB plus = pred, minus = pred;
        plus(i).stress(x, y) += h;
        minus(i).stress(x, y) -= h;
        if (x != y) {
          plus(i).stress(y, x) += h;
          minus(i).stress(y, x) -= h;
        }
        const double fd = (mean_rmse(plus) - mean_rmse(minus)) / (2.0 * h);
        INFO("structure " << i << " entry " << x << y);
        CHECK(g.at({i, comp}) == Approx(fd).epsilon(1e-6).margin(1e-6));
      }
  }
}

/* Pins: tadah predict -S -A prints the stress metrics of the nine entries
 * in GPa: with a potential whose weights are 0 (every prediction 0), the
 * Stress RMSE and MAE of the one dataset are those of the reference
 * tensors themselves, sqrt(sum_ab S_ab^2 / (9 n)) / kGPaInEvPerA3 and
 * sum_ab |S_ab| / (9 n) / kGPaInEvPerA3, and its R^2 is 1 - sum S^2 /
 * sum (S - mean)^2 over the nine entries.
 * Before the fix: the lines read six entries in eV/A^3. */
TEST_CASE("tadah predict -A: the stress metrics are in GPa over the nine "
          "entries",
          "[analytics][stress_gpa][predict]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  StructureDB db;
  const double S[2][6] = {{0.010, 0.002, -0.001, 0.020, 0.003, 0.030},
                          {-0.004, 0.0, 0.005, 0.006, -0.002, 0.001}};
  for (int s = 0; s < 2; ++s) {
    Structure st;
    st.label = "cell " + std::to_string(s);
    const double a = 3.3 + 0.1 * s;
    st.cell.load(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("Ta"), 0.5 * a, 0.5 * a, 0.5 * a, 0.0, 0.0, 0.0));
    st.energy = -20.0;
    set_stress(st, S[s][0], S[s][1], S[s][2], S[s][3], S[s][4], S[s][5]);
    db.add(st);
  }
  db.dump_to_file("db.tadah", false);
  Context pot;
  pot.add("ATOM", "Ta");
  pot.add("WATOM", 73.0);
  pot.add("INIT2B", "true");
  pot.add("INITMB", "false");
  pot.add("TYPE2B", "D2_Blip");
  pot.add("TYPE2B", 2);
  pot.add("TYPE2B", 2);
  pot.add("TYPE2B", "Ta");
  pot.add("TYPE2B", "Ta");
  pot.add("CGRID2B", 2.8);
  pot.add("CGRID2B", 3.5);
  pot.add("SGRID2B", 2.0);
  pot.add("SGRID2B", 2.0);
  pot.add("RCUT2B", 5.0);
  pot.add("RCTYPE2B", "Cut_Cos");
  pot.add("BIAS", "false");
  pot.add("NORM", "false");
  pot.add("MODEL", "M_BLR");
  pot.add("MODEL", "BF_Linear");
  pot.add("WEIGHTS", 0.0);
  pot.add("WEIGHTS", 0.0);
  tadah::mlip::engines::TrainEngine::dumpPotFile("pot.tadah", pot);

  const CommandInput input("PREDICT", {{"potential", {"pot.tadah"}},
                                       {"dbfile", {"db.tadah"}},
                                       {"stress", {"true"}},
                                       {"analytics", {"true"}}});
  std::ostringstream captured;
  std::streambuf *old = std::cout.rdbuf(captured.rdbuf());
  try {
    for (tadah::mlip::Task &t :
         tadah::mlip::TaskFactory().createTasksFromInput(input))
      tadah::mlip::engines::PredictEngine().run(t);
  } catch (...) {
    std::cout.rdbuf(old);
    throw;
  }
  std::cout.rdbuf(old);
  const std::string out = captured.str();
  INFO(out);

  double sq = 0.0, ab = 0.0;
  std::vector<double> obs;
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 6; ++k) {
      sq += kM[k] * S[s][k] * S[s][k];
      ab += kM[k] * std::fabs(S[s][k]);
    }
  for (std::size_t s = 0; s < 2; ++s)
    for (double v : nine(db(s))) obs.push_back(v);
  const std::vector<double> zero(obs.size(), 0.0);
  const std::vector<double> rmse = metric_values(out, "Stress RMSE");
  const std::vector<double> mae = metric_values(out, "Stress MAE");
  const std::vector<double> rsq = metric_values(out, "Stress R^2");
  REQUIRE(rmse.size() == 2);  // the mean and the one dataset
  REQUIRE(mae.size() == 2);
  REQUIRE(rsq.size() == 2);
  // predict -A prints 7 significant digits
  CHECK(rmse[1] == Approx(kC * std::sqrt(sq / 18.0)).epsilon(1e-6));
  CHECK(mae[1] == Approx(kC * ab / 18.0).epsilon(1e-6));
  CHECK(rsq[1] == Approx(r_sq(obs, zero)).epsilon(1e-6));
}

/* Pins: the starter target of write_hpotarget (tadah refit and refit
 * --edit) weights the validation RMSEs with 1/a^2 for the accuracy a of
 * each quantity's default error scale: PC_ERMSE 0 1 (1 meV/atom), the
 * commented PC_FRMSE 0 100 (0.1 eV/A) and PC_SRMSE_GPA 0 4 (0.5 GPa).
 * Before the fix: PC_ERMSE 0 1 and a commented PC_FRMSE 0 1, no stress
 * line. */
TEST_CASE("write_hpotarget: the starter target weights the RMSEs by the "
          "default error scales",
          "[stress_gpa][hpotarget_writer]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  Context c;
  c.add("ATOM", "Cu");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "Cu");
  c.add("TYPE2B", "Cu");
  c.add("CGRID2B", 3.0);
  c.add("CGRID2B", 4.0);
  c.add("SGRID2B", 1.0);
  c.add("SGRID2B", 2.0);
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("FIXWEIGHT", 1.5);
  c.add("FIXWEIGHT", -2.0);
  tadah::mlip::engines::write_hpotarget("t.hpotarget", c, 5.0);
  std::vector<std::string> lines;
  {
    std::istringstream in(slurp("t.hpotarget"));
    for (std::string l; std::getline(in, l);) lines.push_back(l);
  }
  auto has = [&](const std::string &l) {
    for (const std::string &x : lines)
      if (x == l) return true;
    return false;
  };
  CHECK(has("PC_ERMSE 0 1"));
  CHECK(has("# PC_FRMSE 0 100"));
  CHECK(has("# PC_SRMSE_GPA 0 4"));
  CHECK_FALSE(has("# PC_FRMSE 0 1"));
  for (const std::string &x : lines) CHECK(x.find("PC_SRMSE ") == std::string::npos);
}
