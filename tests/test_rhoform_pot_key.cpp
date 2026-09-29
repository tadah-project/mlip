// test_rhoform_pot_key.cpp
//
// RHOFORM (DM_REAM density form, FEAT-02) at the MLIP level:
//  - every potential written for a DM_REAM model carries the declared form,
//    canonicalised to upper case (M_BLR::get_param_file, dumpPotFile);
//    a DM_REAM model without the key is refused (nothing supplies it), and
//    potentials without a DM_REAM block never carry it;
//  - the predict-side precedence between a loaded potential and the task:
//    the potential must declare its own form, whatever the task says.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/analysis_bfunc_engine.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/descriptors/dm/rhoform.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Trainer;
using tadah::mlip::engines::TrainEngine;
using tadah::mlip::memory::MLIPWorkspaceManager;
using tadah::models::DensityFormError;
namespace fs = std::filesystem;

/* D2_Blip (3) + DM_REAM Blip (3 density, 4 embedding), non-negative
 * amplitudes (valid in every form); `form` empty = no RHOFORM line. */
Context make_ream_context(const std::string &form) {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 3);
  c.add("TYPE2B", 3);
  c.add("TYPE2B", "H");
  c.add("TYPE2B", "H");
  c.add("CGRID2B", "1.2 2.2 3.2");
  c.add("SGRID2B", "1.0 1.0 1.0");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 0);
  c.add("TYPEMB", 3); c.add("TYPEMB", 3); c.add("TYPEMB", 3);
  c.add("TYPEMB", 4); c.add("TYPEMB", 4);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  c.add("CGRIDMB", "1.0 2.0 3.0");
  c.add("SGRIDMB", "1.0 1.0 1.0");
  c.add("AMPGRIDMB", "0.40 0.25 0.12");
  if (!form.empty()) c.add("RHOFORM", form);
  c.add("CEMBFUNC", "0.0 0.5 1.0 1.5");
  c.add("SEMBFUNC", "1.5 1.5 1.5 1.5");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 4.5);
  c.add("RCUTMB", 4.5);
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

/* D2_Blip only (INITMB false): a potential without a DM_REAM block. */
Context make_pair_context() {
  Context c = make_ream_context("");
  c.remove("INITMB");
  c.add("INITMB", "false");
  c.remove("TYPEMB");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  for (const char *k : {"CGRIDMB", "SGRIDMB", "AMPGRIDMB", "CEMBFUNC",
                        "SEMBFUNC"})
    c.remove(k);
  return c;
}

void pin_all(Context &c, std::size_t K) {
  c.add("FIXINDEX", "1-" + std::to_string(K));
  for (std::size_t k = 0; k < K; ++k) c.add("FIXWEIGHT", 0.1 * k - 0.2);
}

std::size_t n_weights(const Context &proto, MLIPWorkspaceManager &ws) {
  Context probe_ctx = proto;
  Trainer probe(probe_ctx, ws);
  return probe.fb->get_phi_cols(probe.context);
}

std::string slurp(const std::string &f) {
  std::ifstream in(f);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

struct ScopedTempDir {
  fs::path old_cwd, dir;
  ScopedTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_rhoform_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScopedTempDir() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

}  // namespace

TEST_CASE("RHOFORM: DM_REAM potentials carry the declared form",
          "[rhoform][pot_key]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  const std::size_t K = n_weights(make_ream_context("SIGNED"), ws);

  SECTION("every form is written, canonical upper case") {
    for (const char *in : {"SIGNED", "nonneg", "Monotone"}) {
      INFO(in);
      Context c = make_ream_context(in);
      pin_all(c, K);
      Trainer tr(c, ws);
      tr.provision_fixed_weights();
      const Context pot = tr.get_param_file();
      const auto form = tadah::models::parse_rhoform(in);
      REQUIRE(pot.stored_value("RHOFORM") ==
              std::optional<std::string>(tadah::models::rhoform_name(form)));
    }
  }
  SECTION("a DM_REAM model without the key is refused, not defaulted") {
    Context c = make_ream_context("");
    REQUIRE_THROWS_AS(Trainer(c, ws), DensityFormError);
    REQUIRE_THROWS_WITH(Trainer(c, ws),
                        Catch::Contains("RHOFORM is required: SIGNED"));
  }
  SECTION("dumpPotFile: written for DM_REAM, refused without the key, "
          "stripped otherwise") {
    ScopedTempDir tmp;
    TrainEngine::dumpPotFile("ream.tadah", make_ream_context("monotone"));
    const Context back =
        tadah::core::CoreUserConfigParser().parse("ream.tadah");
    REQUIRE(back.stored_value("RHOFORM") ==
            std::optional<std::string>("MONOTONE"));

    REQUIRE_THROWS_AS(
        TrainEngine::dumpPotFile("keyless.tadah", make_ream_context("")),
        DensityFormError);
    REQUIRE_FALSE(fs::exists("keyless.tadah"));  // nothing is written

    Context pair = make_pair_context();
    pair.add("RHOFORM", "SIGNED");
    TrainEngine::dumpPotFile("pair.tadah", pair);
    REQUIRE(slurp("pair.tadah").find("RHOFORM") == std::string::npos);
  }
  SECTION("get_param_file strips a stray key without DM_REAM") {
    const std::size_t Kp = n_weights(make_pair_context(), ws);
    Context stray = make_pair_context();
    pin_all(stray, Kp);
    stray.add("RHOFORM", "NONNEG");
    Trainer tr(stray, ws);
    tr.provision_fixed_weights();
    REQUIRE_FALSE(tr.get_param_file().has_stored("RHOFORM"));
  }
}

TEST_CASE("RHOFORM: potential vs task precedence", "[rhoform][precedence]") {
  using tadah::models::check_rhoform_precedence;
  Context task;
  SECTION("the potential's form, no task key: accepted") {
    REQUIRE_NOTHROW(check_rhoform_precedence(make_ream_context("SIGNED"), task,
                                             "pot.tadah"));
  }
  SECTION("the same form in the task (any case): accepted") {
    task.add("RHOFORM", "signed");
    REQUIRE_NOTHROW(check_rhoform_precedence(make_ream_context("SIGNED"), task,
                                             "pot.tadah"));
  }
  SECTION("a different form in the task: refused") {
    task.add("RHOFORM", "NONNEG");
    REQUIRE_THROWS_AS(check_rhoform_precedence(make_ream_context("SIGNED"),
                                               task, "pot.tadah"),
                      DensityFormError);
    REQUIRE_THROWS_WITH(check_rhoform_precedence(make_ream_context("SIGNED"),
                                                 task, "pot.tadah"),
                        Catch::Contains("RHOFORM conflict: pot.tadah "
                                        "declares RHOFORM SIGNED"));
  }
  SECTION("a key-less DM_REAM potential: refused even with a task key") {
    task.add("RHOFORM", "SIGNED");
    REQUIRE_THROWS_AS(
        check_rhoform_precedence(make_ream_context(""), task, "old.tadah"),
        DensityFormError);
    REQUIRE_THROWS_WITH(
        check_rhoform_precedence(make_ream_context(""), task, "old.tadah"),
        Catch::Contains("old.tadah: the task configuration's RHOFORM does "
                        "not apply") &&
            Catch::Contains("DM_REAM: RHOFORM is required"));
    REQUIRE_THROWS_AS(check_rhoform_precedence(make_ream_context(""),
                                               Context(), "old.tadah"),
                      DensityFormError);
  }
  SECTION("no DM_REAM block: a key on either side is ignored") {
    Context pair = make_pair_context();
    pair.add("RHOFORM", "SIGNED");
    task.add("RHOFORM", "MONOTONE");
    REQUIRE_NOTHROW(check_rhoform_precedence(pair, task, "pair.tadah"));
  }
}

TEST_CASE("RHOFORM: tadah predict refuses a key-less DM_REAM potential "
          "before the task is merged",
          "[rhoform][precedence][predict]") {
  PeriodicTable::initialize();
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("pot.tadah", make_ream_context("SIGNED"));
  {  // the same potential as tadah wrote it before 1.6.0: no RHOFORM line
    std::istringstream in(slurp("pot.tadah"));
    std::ofstream out("old.tadah");
    for (std::string line; std::getline(in, line);)
      if (line.rfind("RHOFORM", 0) != 0) out << line << "\n";
  }
  Context c;
  c.add("POTENTIAL", std::string("old.tadah"));
  c.add("RHOFORM", std::string("SIGNED"));  // the task cannot supply it
  tadah::mlip::Task task("predict", c);
  tadah::mlip::engines::PredictEngine engine;
  REQUIRE_THROWS_AS(engine.run(task), DensityFormError);
  REQUIRE_THROWS_WITH(engine.run(task),
                      Catch::Contains("old.tadah: the task configuration's "
                                      "RHOFORM does not apply") &&
                          Catch::Contains("DM_REAM: RHOFORM is required"));
}

TEST_CASE("RHOFORM and EMBF0: tadah analysis bfunc applies the "
          "potential-versus-task precedence of predict",
          "[rhoform][embf0][precedence][bfunc]") {
  // analysis bfunc merged the task over the potential without the checks
  // of predict and analysis descriptor: a task RHOFORM replaced the
  // potential's form, and supplied one to a key-less DM_REAM potential.
  PeriodicTable::initialize();
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("pot.tadah", make_ream_context("SIGNED"));
  {  // the same potential as tadah wrote it before 1.6.0: no RHOFORM line
    std::istringstream in(slurp("pot.tadah"));
    std::ofstream out("old.tadah");
    for (std::string line; std::getline(in, line);)
      if (line.rfind("RHOFORM", 0) != 0) out << line << "\n";
  }
  auto bfunc = [](const char *pot, const char *key, const char *value) {
    Context c;
    c.add("POTENTIAL", std::string(pot));
    c.add("TYPE", std::string("MB"));
    c.add("RANGE", std::string("0.5 4.0 8"));
    c.add("OUTFILE", std::string("bfunc.txt"));
    if (key) c.add(key, std::string(value));
    return tadah::mlip::Task("analysis", c);
  };
  tadah::mlip::engines::AnalysisBfuncEngine engine;
  {
    auto t = bfunc("pot.tadah", "RHOFORM", "NONNEG");
    REQUIRE_THROWS_AS(engine.run(t), DensityFormError);
    REQUIRE_THROWS_WITH(engine.run(t),
                        Catch::Contains("RHOFORM conflict: pot.tadah declares "
                                        "RHOFORM SIGNED but the task "
                                        "configuration sets RHOFORM NONNEG"));
  }
  {
    auto t = bfunc("old.tadah", "RHOFORM", "SIGNED");
    REQUIRE_THROWS_WITH(engine.run(t),
                        Catch::Contains("old.tadah: the task configuration's "
                                        "RHOFORM does not apply"));
  }
  {
    auto t = bfunc("pot.tadah", "EMBF0", "FREE");
    REQUIRE_THROWS_WITH(engine.run(t),
                        Catch::Contains("EMBF0 conflict: pot.tadah declares "
                                        "EMBF0 ZERO but the task "
                                        "configuration sets EMBF0 FREE"));
  }
  REQUIRE_FALSE(fs::exists("bfunc.txt"));
  // the potential's own values in the task: the analysis runs
  auto t = bfunc("pot.tadah", "RHOFORM", "signed");
  REQUIRE(engine.run(t) == EXIT_SUCCESS);
  REQUIRE(fs::exists("bfunc.txt"));
}
