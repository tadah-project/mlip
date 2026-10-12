// test_m_krr_param_file.cpp
//
// M_KRR::get_param_file() -- the context `tadah train` writes to pot.tadah
// and `tadah hpo` writes to best_pot.tadah -- keeps the training FORCE and
// STRESS, as M_BLR::get_param_file() does, so that the potential states what
// it was fitted to and `tadah train` with the potential's keys reproduces the
// fit. It still drops ALPHA, BETA (the evidence-approximation starting
// guesses) and DBFILE (the training data).
//
// Before the fix M_KRR::get_param_file() removed FORCE and STRESS as well:
// a potential fitted to forces said nothing about it, and HPO's validation of
// an M_KRR candidate read FORCE/STRESS from that context and got the
// blueprint default (false).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/models/m_blr.h>
#include <tadah/mlip/models/m_krr.h>
#include <tadah/mlip/trainer.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
namespace fs = std::filesystem;

/* A D2_Blip pair potential (INITMB false) with the given MODEL tokens and
 * explicit FORCE/STRESS, and the keys get_param_file() must drop: ALPHA,
 * BETA and DBFILE (the training file is never opened here). */
Context make_context(const std::vector<std::string> &model, bool force,
                     bool stress) {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 3);
  c.add("TYPE2B", 3);
  c.add("TYPE2B", "H");
  c.add("TYPE2B", "H");
  c.add("CGRID2B", "1.2 2.2 3.2");
  c.add("SGRID2B", "1.0 1.0 1.0");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 4.5);
  c.add("RCUTMB", 4.5);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", force ? "true" : "false");
  c.add("STRESS", stress ? "true" : "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  for (const auto &m : model) c.add("MODEL", m);
  c.add("LAMBDA", 0);
  c.add("ALPHA", 0.5);
  c.add("BETA", 2.0);
  c.add("DBFILE", "kkrr_train.db");
  return c;
}

const std::vector<std::string> kKRR = {"M_KRR", "Kern_Linear", "1"};
const std::vector<std::string> kBLR = {"M_BLR", "BF_Linear"};

/* Pins every weight (FIXINDEX/FIXWEIGHT), so that the model is a complete
 * potential without training data. */
void pin_all_weights(Context &c, MLIPWorkspaceManager &ws) {
  Context probe_ctx = c;
  Trainer probe(probe_ctx, ws);
  const std::size_t K = probe.fb->get_phi_cols(probe.context);
  REQUIRE(K > 0);
  c.add("FIXINDEX", "1-" + std::to_string(K));
  for (std::size_t k = 0; k < K; ++k) c.add("FIXWEIGHT", 0.1 * (k + 1));
}

/* KEY -> tokens of a written potential's lines (column spacing ignored). */
std::map<std::string, std::vector<std::string>>
read_pot_lines(const std::string &path) {
  std::map<std::string, std::vector<std::string>> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string key, tok;
    if (!(ss >> key)) continue;
    auto &v = out[key];
    while (ss >> tok) v.push_back(tok);
  }
  return out;
}

/* A stored token, or "<absent>": readable in a failure, and never inserts
 * the blueprint default (get<bool>() would turn an absent STRESS into
 * false and pass the "false" half by accident). */
const std::string kAbsent = "<absent>";
std::string stored(const Context &c, const std::string &key,
                   std::size_t n = 0) {
  return c.stored_value(key, n).value_or(kAbsent);
}

} // namespace

/* Contract: an M_KRR (Kern_Linear) potential stores the training FORCE and
 * STRESS verbatim -- both FORCE true / STRESS false and FORCE false / STRESS
 * true -- and none of ALPHA, BETA, DBFILE; the written pot.tadah has FORCE
 * and STRESS lines and no ALPHA/BETA/DBFILE line.
 *
 * Fails before the fix: M_KRR::get_param_file() calls c.remove("FORCE") and
 * c.remove("STRESS"), so FORCE and STRESS come back "<absent>" (for both
 * combinations) and the written file has no FORCE or STRESS line. (The
 * checks read stored values on purpose: get<bool>() would insert the
 * blueprint default false and pass the "false" half by accident.) */
TEST_CASE("M_KRR::get_param_file keeps the training FORCE and STRESS and "
          "drops ALPHA, BETA and DBFILE",
          "[m_krr][param_file]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;

  struct Switches { bool force, stress; };
  const Switches sw = GENERATE(Switches{true, false}, Switches{false, true});
  const std::string want_f = sw.force ? "true" : "false";
  const std::string want_s = sw.stress ? "true" : "false";
  INFO("training FORCE " << want_f << ", STRESS " << want_s);

  Context c = make_context(kKRR, sw.force, sw.stress);
  pin_all_weights(c, ws);
  Trainer tr(c, ws);
  REQUIRE(dynamic_cast<tadah::mlip::M_KRR<> *>(tr.model) != nullptr);
  tr.provision_fixed_weights();

  const Context pot = tr.model->get_param_file();

  // The model and its weights are written as before.
  REQUIRE(stored(pot, "MODEL", 0) == "M_KRR");
  REQUIRE(stored(pot, "MODEL", 1) == "Kern_Linear");
  REQUIRE(stored(pot, "MODEL", 2) == "1");
  REQUIRE(pot.has_stored("WEIGHTS"));

  // The training FORCE/STRESS, verbatim (one token each).
  CHECK(stored(pot, "FORCE") == want_f);
  CHECK(stored(pot, "STRESS") == want_s);
  CHECK(stored(pot, "FORCE", 1) == kAbsent);
  CHECK(stored(pot, "STRESS", 1) == kAbsent);

  // Still dropped.
  CHECK_FALSE(pot.has_stored("ALPHA"));
  CHECK_FALSE(pot.has_stored("BETA"));
  CHECK_FALSE(pot.has_stored("DBFILE"));

  // Trainer::get_param_file() (what TrainEngine and HPO write) is the same.
  CHECK(stored(tr.get_param_file(), "FORCE") == want_f);
  CHECK(stored(tr.get_param_file(), "STRESS") == want_s);

  // The model's own context still holds what it was given.
  CHECK(stored(tr.context, "FORCE") == want_f);
  CHECK(stored(tr.context, "STRESS") == want_s);

  // The written potential (TrainEngine::dumpPotFile, as `tadah train` and
  // best_pot.tadah) has the FORCE and STRESS lines and none of the others.
  const fs::path dir = fs::temp_directory_path() /
                       ("tadah_krr_param_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  const std::string f =
      (dir / ("pot_" + want_f + "_" + want_s + ".tadah")).string();
  tadah::mlip::engines::TrainEngine::dumpPotFile(f, pot);
  const auto lines = read_pot_lines(f);
  fs::remove_all(dir);

  REQUIRE(lines.count("MODEL") == 1);
  CHECK(lines.count("FORCE") == 1);
  CHECK(lines.count("STRESS") == 1);
  if (lines.count("FORCE"))
    CHECK(lines.at("FORCE") == std::vector<std::string>{want_f});
  if (lines.count("STRESS"))
    CHECK(lines.at("STRESS") == std::vector<std::string>{want_s});
  CHECK(lines.count("ALPHA") == 0);
  CHECK(lines.count("BETA") == 0);
  CHECK(lines.count("DBFILE") == 0);
}

/* M_BLR, the reference M_KRR now follows: its potential keeps FORCE and
 * STRESS verbatim. Pins existing behaviour; fails if the commented-out
 * c.remove("FORCE") / c.remove("STRESS") in M_BLR::get_param_file()
 * (m_blr.h) are reinstated. */
TEST_CASE("M_BLR::get_param_file keeps the training FORCE and STRESS",
          "[m_blr][param_file]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;

  struct Switches { bool force, stress; };
  const Switches sw = GENERATE(Switches{true, false}, Switches{false, true});
  const std::string want_f = sw.force ? "true" : "false";
  const std::string want_s = sw.stress ? "true" : "false";
  INFO("training FORCE " << want_f << ", STRESS " << want_s);

  Context c = make_context(kBLR, sw.force, sw.stress);
  pin_all_weights(c, ws);
  Trainer tr(c, ws);
  REQUIRE(dynamic_cast<tadah::mlip::M_BLR<> *>(tr.model) != nullptr);
  tr.provision_fixed_weights();

  const Context pot = tr.model->get_param_file();
  REQUIRE(stored(pot, "MODEL", 0) == "M_BLR");
  CHECK(stored(pot, "FORCE") == want_f);
  CHECK(stored(pot, "STRESS") == want_s);
}

/* Pins: neither M_KRR's nor M_BLR's potential records UNCERTAINTY, the
 * switch of the train task that writes weights.tadah (tadah train
 * --uncertainty): with UNCERTAINTY true in the model's context,
 * get_param_file() has no UNCERTAINTY, nor has the potential
 * TrainEngine::dumpPotFile writes from it.
 * Before the fix: both potentials had UNCERTAINTY true (every potential
 * carried the run's UNCERTAINTY), and `tadah train -c <potential>` read
 * it as a request: a retrain with another OALGO was refused ("tadah train
 * --uncertainty needs OALGO 3") with no --uncertainty given. */
TEST_CASE("get_param_file does not record UNCERTAINTY", "[param_file]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  for (const auto &model : {kKRR, kBLR}) {
    DYNAMIC_SECTION(model[0]) {
      Context c = make_context(model, true, false);
      c.add("UNCERTAINTY", "true");
      pin_all_weights(c, ws);
      Trainer tr(c, ws);
      tr.provision_fixed_weights();
      REQUIRE(stored(tr.context, "UNCERTAINTY") == "true");
      const Context pot = tr.model->get_param_file();
      CHECK_FALSE(pot.has_stored("UNCERTAINTY"));
      const fs::path dir = fs::temp_directory_path() /
                           ("tadah_unc_param_" + std::to_string(::getpid()));
      fs::create_directories(dir);
      const std::string f = (dir / (model[0] + ".tadah")).string();
      tadah::mlip::engines::TrainEngine::dumpPotFile(f, pot);
      const auto lines = read_pot_lines(f);
      fs::remove_all(dir);
      CHECK(lines.count("WEIGHTS") == 1);
      CHECK(lines.count("UNCERTAINTY") == 0);
    }
  }
}
