// A potential with a knot descriptor of a D2_mJoin or DM_mJoin line is
// evaluated at the cutoffs it states, with which its coefficients were
// fitted: tadah predict and tadah refit --edit keep them, where a fit
// (tadah train, tadah hpo) derives the cutoff of such a descriptor from its
// largest knot.
//
// tests_data/potentials_1_5/joined_knot/ holds two fits of tadah 1.5.0,
// which fitted a joined line at its stated cutoffs and wrote them:
// eos8_333.tadahdb (the 3x3x3 supercells, 54 atoms each, of the first 8
// structures of the bcc Ta energy-volume set of HPO's
// tests/data/ta/common/data/eos_bcc.tadahdb, energies only: a cell of
// 7.8 A or more in each direction, of which tadah 1.5.0 listed every
// neighbour within the cutoffs, as tadah does), and the potentials
// `tadah train -c <config> -o <pot>` of tadah 1.5.0 wrote from
//  - config_2b.train: TYPE2B D2_mJoin of D2_Knot5 3 3 Ta Ta (knots 2.5 3.5
//    4.5) and D2_Blip 3 3 Ta Ta, RCTYPE2B Cut_Cos Cut_Cos, RCUT2B 5.3 5.0,
//    M_BLR BF_Linear, OALGO 1 1e-13: pot_2b_1.5.0.tadah (RCUT2B 5.3 5);
//  - config_mb.train: TYPE2B D2_Blip 3 3 Ta Ta, RCUT2B 5, and TYPEMB
//    DM_mJoin of DM_Blip 0 2 2 Ta Ta and a knot DM_REAM 2 2 0 2 3 0 0 0 Ta
//    Ta (knots 4.6 3.0), RCTYPEMB Cut_Cos Cut_Cos, RCUTMB 5.0 5.2:
//    pot_mb_1.5.0.tadah (RCUTMB 5 5.2; 1.5.0 had no RHOFORM, so the cases
//    add the line RHOFORM SIGNED, the density of every tadah before 1.6.0,
//    as tadah asks of such a potential).
// The energies per atom tadah 1.5.0 predicted for eos8_333.tadahdb with
// each potential (`tadah predict -p <pot> -d eos8_333.tadahdb`) are kOld2B
// and kOldMB below.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/key_notice.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::IgnoredKeyNoticeCapture;
using tadah::core::PeriodicTable;
using tadah::mlip::Task;
namespace fs = std::filesystem;

const fs::path kDir = fs::absolute("tests_data/potentials_1_5/joined_knot");

/* The energies per atom tadah 1.5.0 predicted for eos8_333.tadahdb. */
const std::vector<double> kOld2B = {
    1.308138951518767135e+00,  2.364384205686111129e-01,
    -7.343011289180407797e-01, -1.613693446083312111e+00,
    -2.409787791523580580e+00, -3.129326230966277489e+00,
    -3.778024378712358633e+00, -4.360927405640062204e+00};
const std::vector<double> kOldMB = {
    1.312856119234951269e+00,  2.226789233973249793e-01,
    -7.300683222657190496e-01, -1.601446295511171458e+00,
    -2.406129328256981559e+00, -3.139583679806727812e+00,
    -3.790787198096007060e+00, -4.348922534762836634e+00};

/* The working directory of a case. */
struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_joined_knot_" + std::to_string(::getpid()) + "_" +
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

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* What TaskFactory gets from the command line. */
struct CommandInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  CommandInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* Runs @p run on each task TaskFactory makes of @p input. */
template <typename Run>
void run_input(const CommandInput &input, Run run) {
  IgnoredKeyNoticeCapture quiet;
  for (Task &t : tadah::mlip::TaskFactory().createTasksFromInput(input))
    run(t);
}

/* The predicted energies of energy.pred (Index, True Energy, Pred Energy
 * after a title line). */
std::vector<double> predicted_energies(const std::string &file) {
  std::ifstream in(file);
  std::string line;
  std::getline(in, line);
  std::vector<double> out;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    double index = 0, truth = 0, pred = 0;
    if (ls >> index >> truth >> pred) out.push_back(pred);
  }
  return out;
}

/* The energies tadah predict gives for eos8_333.tadahdb with @p pot. */
std::vector<double> predict(const std::string &pot) {
  run_input(CommandInput("PREDICT",
                         {{"potential", {pot}},
                          {"dbfile", {(kDir / "eos8_333.tadahdb").string()}}}),
            [](Task &t) { tadah::mlip::engines::PredictEngine().run(t); });
  return predicted_energies("energy.pred");
}

/* The largest difference of @p got from @p ref, entry by entry. */
double largest_difference(const std::vector<double> &got,
                          const std::vector<double> &ref) {
  REQUIRE(got.size() == ref.size());
  double diff = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i)
    diff = std::max(diff, std::fabs(got[i] - ref[i]));
  return diff;
}

/* The values of @p key of the potential file @p pot. */
std::vector<double> values_of(const std::string &pot, const std::string &key) {
  IgnoredKeyNoticeCapture quiet;
  const Context c = tadah::core::CoreUserConfigParser().parse(pot);
  std::vector<double> v(c.size(key));
  c.get(key, v);
  return v;
}

}  // namespace

/* Pins: tadah predict evaluates a potential of a D2_mJoin line of
 * D2_Knot5 (knots up to 4.5) and D2_Blip that states RCUT2B 5.3 5 under
 * Cut_Cos at those cutoffs, with which tadah 1.5.0 fitted its coefficients:
 * its energies per atom of eos8_333.tadahdb are those tadah 1.5.0
 * predicted (kOld2B), to 1e-9 eV.
 * Before the fix: predict re-derived the D2_Knot5's cutoff from its
 * largest knot (RCUT2B 4.5 5) and evaluated the coefficients fitted at 5.3
 * at 4.5: the energies were off by up to 0.91 eV per atom. */
TEST_CASE("tadah predict evaluates a potential with a joined D2_Knot5 at "
          "its stated cutoffs",
          "[join][knot][predict]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  const std::vector<double> e =
      predict((kDir / "pot_2b_1.5.0.tadah").string());
  CHECK(largest_difference(e, kOld2B) < 1e-9);
}

/* Pins: tadah refit --edit (FUNC phi, ADD 3.7: one basis function of
 * coefficient 0) of a potential whose TYPEMB DM_mJoin line has a knot
 * DM_REAM density (knots 4.6 3.0) that states RCUTMB 5.2 (RCUTMB 5 5.2,
 * Cut_Cos) writes RCUTMB 5 5.2 as stated, and the edited potential, as
 * the input, predicts the energies tadah 1.5.0 predicted with the input
 * (kOldMB), to 1e-9 eV per atom: adding a basis function of coefficient 0
 * leaves the potential's energies as they are.
 * Before the fix: refit --edit re-derived the DM_REAM's cutoff and wrote
 * RCUTMB 5 4.6 with the coefficients fitted at 5.2, and predict
 * re-derived it in both potentials: their energies were off by up to
 * 3.59 eV per atom. */
TEST_CASE("tadah refit --edit keeps the stated cutoffs of a joined knot "
          "DM_REAM density",
          "[join][knot][refit][edit]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  {
    std::ifstream in(kDir / "pot_mb_1.5.0.tadah");
    std::ofstream out("in.tadah");
    out << in.rdbuf() << "RHOFORM SIGNED\n";
  }
  {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.7);
    Task task("refit", c);
    IgnoredKeyNoticeCapture quiet;
    REQUIRE(tadah::mlip::engines::RefitEngine().run(task) == EXIT_SUCCESS);
  }
  const std::vector<double> rcut = values_of("pot_edited.tadah", "RCUTMB");
  REQUIRE(rcut.size() == 2u);
  CHECK(rcut[0] == 5.0);
  CHECK(rcut[1] == 5.2);
  CHECK(largest_difference(predict("in.tadah"), kOldMB) < 1e-9);
  CHECK(largest_difference(predict("pot_edited.tadah"), kOldMB) < 1e-9);
}
