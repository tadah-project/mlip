// test_m_krr_param_file.cpp
//
// M_KRR::get_param_file() and M_BLR::get_param_file() -- the context `tadah
// train` writes to pot.tadah and `tadah hpo` writes to best_pot.tadah --
// follow one rule: a potential records the model's settings and its fit,
// not the training data. Both keep the training FORCE and STRESS, so that
// the potential states what it was fitted to and `tadah train` with the
// potential's keys reproduces the fit, and ALPHA and BETA (the evidence
// approximation's starting guesses, or after a LAMBDA -1 fit the values it
// found); neither writes DBFILE (the training data), so a potential retrains
// with the DBFILE lines of the retrain added (`tadah train -c pot.tadah`
// plus a DBFILE), whichever the model.
//
// Before the FORCE/STRESS fix M_KRR::get_param_file() removed FORCE and
// STRESS: a potential fitted to forces said nothing about it, and HPO's
// validation of an M_KRR candidate read FORCE/STRESS from that context and
// got the blueprint default (false). Before the potential-content fix
// M_KRR::get_param_file() removed ALPHA and BETA, and M_BLR's kept DBFILE.

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
 * explicit FORCE/STRESS, ALPHA 0.5 and BETA 2.0, and DBFILE, which
 * get_param_file() must drop (the training file is never opened here). */
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

/* Pins: an M_KRR (Kern_Linear) potential stores the training FORCE and
 * STRESS verbatim -- both FORCE true / STRESS false and FORCE false / STRESS
 * true -- and ALPHA 0.5 and BETA 2.0 of the configuration, and no DBFILE;
 * the written pot.tadah has FORCE, STRESS, ALPHA and BETA lines and no
 * DBFILE line.
 *
 * Before the FORCE/STRESS fix: M_KRR::get_param_file() called
 * c.remove("FORCE") and c.remove("STRESS"), so FORCE and STRESS came back
 * "<absent>" (for both combinations) and the written file had no FORCE or
 * STRESS line. (The checks read stored values on purpose: get<bool>() would
 * insert the blueprint default false and pass the "false" half by
 * accident.)
 * Before the potential-content fix: ALPHA and BETA came back "<absent>"
 * (M_KRR::get_param_file() removed them; this case checked that they were
 * dropped), and the written file had no ALPHA or BETA line. */
TEST_CASE("M_KRR::get_param_file keeps the training FORCE, STRESS, ALPHA "
          "and BETA and drops DBFILE",
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

  // The evidence approximation's ALPHA and BETA, as configured; no DBFILE.
  CHECK(stored(pot, "ALPHA") == "0.5");
  CHECK(stored(pot, "BETA") == "2");
  CHECK_FALSE(pot.has_stored("DBFILE"));

  // Trainer::get_param_file() (what TrainEngine and HPO write) is the same.
  CHECK(stored(tr.get_param_file(), "FORCE") == want_f);
  CHECK(stored(tr.get_param_file(), "STRESS") == want_s);

  // The model's own context still holds what it was given.
  CHECK(stored(tr.context, "FORCE") == want_f);
  CHECK(stored(tr.context, "STRESS") == want_s);

  // The written potential (TrainEngine::dumpPotFile, as `tadah train` and
  // best_pot.tadah) has the FORCE, STRESS, ALPHA and BETA lines and no
  // DBFILE line.
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
  CHECK(lines.count("ALPHA") == 1);
  CHECK(lines.count("BETA") == 1);
  CHECK(lines.count("DBFILE") == 0);
}

/* Pins: an M_BLR (BF_Linear) potential, by the rule M_KRR's follows,
 * stores the training FORCE and STRESS verbatim (both combinations), ALPHA
 * 0.5 and BETA 2.0 of the configuration, and no DBFILE, and so does the
 * pot.tadah TrainEngine::dumpPotFile writes from it.
 * Before the potential-content fix: the potential kept DBFILE
 * (kkrr_train.db; m_blr.h had its c.remove("DBFILE") commented out), so
 * `tadah train -c pot.tadah` with the documented DBFILE line added listed
 * the dataset twice and fitted every structure twice. FORCE, STRESS, ALPHA
 * and BETA were kept then as well (a pin of those). */
TEST_CASE("M_BLR::get_param_file keeps the training FORCE, STRESS, ALPHA "
          "and BETA and drops DBFILE",
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
  CHECK(stored(pot, "ALPHA") == "0.5");
  CHECK(stored(pot, "BETA") == "2");
  CHECK(stored(pot, "DBFILE") == kAbsent);

  const fs::path dir = fs::temp_directory_path() /
                       ("tadah_blr_param_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  const std::string f =
      (dir / ("pot_" + want_f + "_" + want_s + ".tadah")).string();
  tadah::mlip::engines::TrainEngine::dumpPotFile(f, pot);
  const auto lines = read_pot_lines(f);
  fs::remove_all(dir);
  REQUIRE(lines.count("MODEL") == 1);
  CHECK(lines.count("ALPHA") == 1);
  CHECK(lines.count("BETA") == 1);
  CHECK(lines.count("DBFILE") == 0);
}
