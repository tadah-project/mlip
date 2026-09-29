// test_embf0_pot_key.cpp
//
// EMBF0 (DM_REAM embedding reference) at the MLIP level:
//  - every potential written for a DM_REAM model carries an explicit EMBF0
//    line (ZERO by default, FREE preserved), including a DM_mJoin parent;
//    potentials without a DM_REAM block never do;
//  - the predict-side precedence between a loaded potential and the task;
//  - least-squares training under EMBF0 ZERO: the isolated atom predicts 0
//    exactly, and with BIAS true the fitted function equals the FREE fit
//    (span{g - g(0), 1} == span{g, 1}).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/models/descriptors/dm/embf0.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
namespace fs = std::filesystem;

constexpr double kRcut = 4.5;

/* D2_Blip (3) + DM_REAM Blip (3 density, 4 embedding; two embedding bumps
 * cover rho = 0, so g(0) != 0 and ZERO differs from FREE). */
Context make_ream_context(bool bias = false) {
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
  c.add("RHOFORM", "SIGNED");  // mandatory for DM_REAM
  c.add("CEMBFUNC", "0.0 0.5 1.0 1.5");
  c.add("SEMBFUNC", "1.5 1.5 1.5 1.5");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", kRcut);
  c.add("RCUTMB", kRcut);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", bias ? "true" : "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

/* D2_BP only (INITMB false): a potential without a DM_REAM block. */
Context make_pair_context() {
  Context c = make_ream_context();
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

std::size_t n_weights(const Context &proto, MLIPWorkspaceManager &ws) {
  Context probe_ctx = proto;
  Trainer probe(probe_ctx, ws);
  return probe.fb->get_phi_cols(probe.context);
}

void pin_all(Context &c, const std::vector<double> &w) {
  c.add("FIXINDEX", "1-" + std::to_string(w.size()));
  for (double v : w) c.add("FIXWEIGHT", v);
}

std::string pot_text(const Context &pot) {
  std::ostringstream os;
  os << pot;
  return os.str();
}

Structure cluster(std::mt19937 &rng, std::size_t n) {
  std::uniform_real_distribution<double> u(0.0, 3.0);
  Structure st;
  st.cell.load(30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0);
  std::vector<std::array<double, 3>> placed;
  while (placed.size() < n) {
    std::array<double, 3> p{10.0 + u(rng), 10.0 + u(rng), 10.0 + u(rng)};
    bool ok = true;
    for (const auto &q : placed) {
      const double d = std::sqrt((p[0] - q[0]) * (p[0] - q[0]) +
                                 (p[1] - q[1]) * (p[1] - q[1]) +
                                 (p[2] - q[2]) * (p[2] - q[2]));
      if (d < 0.9) ok = false;
    }
    if (ok) placed.push_back(p);
  }
  for (const auto &p : placed)
    st.add_atom(Atom(Element("H"), p[0], p[1], p[2], 0.0, 0.0, 0.0));
  return st;
}

Structure isolated_atom() {
  Structure st;
  st.cell.load(30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0);
  st.add_atom(Atom(Element("H"), 15.0, 15.0, 15.0, 0.0, 0.0, 0.0));
  return st;
}

double energy_at(Trainer &tr, Structure st) {
  NNFinder finder;
  finder.calc(st, kRcut);
  StDescriptors d = tr.dc.calc(st);
  return tr.model->epredict(d);
}

}  // namespace

TEST_CASE("EMBF0: DM_REAM potentials always carry the key",
          "[embf0][pot_key]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  const std::size_t K = n_weights(make_ream_context(), ws);
  std::vector<double> w(K);
  for (std::size_t k = 0; k < K; ++k) w[k] = 0.3 * (k % 5) - 0.4;

  SECTION("absent key -> written as ZERO") {
    Context c = make_ream_context();
    pin_all(c, w);
    REQUIRE_FALSE(c.has_stored("EMBF0"));
    Trainer tr(c, ws);
    tr.provision_fixed_weights();
    const Context pot = tr.get_param_file();
    REQUIRE(pot.stored_value("EMBF0") == std::optional<std::string>("ZERO"));
    REQUIRE(pot_text(pot).find("EMBF0") != std::string::npos);
  }
  SECTION("FREE is preserved, lower case is canonicalised") {
    Context c = make_ream_context();
    pin_all(c, w);
    c.add("EMBF0", "free");
    Trainer tr(c, ws);
    tr.provision_fixed_weights();
    REQUIRE(tr.get_param_file().stored_value("EMBF0") ==
            std::optional<std::string>("FREE"));
  }
  SECTION("an invalid value is refused") {
    Context c = make_ream_context();
    c.add("EMBF0", "ON");
    REQUIRE_THROWS_WITH(Trainer(c, ws), Catch::Contains("ZERO or FREE"));
  }
}

TEST_CASE("EMBF0: potentials without DM_REAM never carry the key",
          "[embf0][pot_key]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  const std::size_t K = n_weights(make_pair_context(), ws);
  std::vector<double> w(K, 0.25);

  Context c = make_pair_context();
  pin_all(c, w);
  Trainer tr(c, ws);
  tr.provision_fixed_weights();
  REQUIRE_FALSE(tr.get_param_file().has_stored("EMBF0"));

  Context stray = make_pair_context();
  pin_all(stray, w);
  stray.add("EMBF0", "FREE");
  Trainer tr2(stray, ws);
  tr2.provision_fixed_weights();
  const Context pot = tr2.get_param_file();
  REQUIRE_FALSE(pot.has_stored("EMBF0"));
  REQUIRE(pot_text(pot).find("EMBF0") == std::string::npos);
}

TEST_CASE("EMBF0: dumpPotFile writes the key for DM_REAM only",
          "[embf0][pot_key]") {
  PeriodicTable::initialize();
  const fs::path dir = fs::temp_directory_path() /
                       ("tadah_embf0_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  const std::string f1 = (dir / "ream.tadah").string();
  const std::string f2 = (dir / "pair.tadah").string();
  tadah::mlip::engines::TrainEngine::dumpPotFile(f1, make_ream_context());
  Context pair = make_pair_context();
  pair.add("EMBF0", "ZERO");
  tadah::mlip::engines::TrainEngine::dumpPotFile(f2, pair);
  auto slurp = [](const std::string &f) {
    std::ifstream in(f);
    return std::string(std::istreambuf_iterator<char>(in), {});
  };
  const std::string s1 = slurp(f1), s2 = slurp(f2);
  std::error_code ec;
  fs::remove_all(dir, ec);
  REQUIRE(s1.find("EMBF0") != std::string::npos);
  REQUIRE(s1.find("ZERO") != std::string::npos);
  REQUIRE(s2.find("EMBF0") == std::string::npos);
}

TEST_CASE("EMBF0: potential vs task precedence", "[embf0][precedence]") {
  using tadah::models::check_embf0_precedence;
  Context pot = make_ream_context();
  Context task;
  SECTION("task without the key: nothing to check") {
    REQUIRE_NOTHROW(check_embf0_precedence(pot, task, "pot.tadah"));
  }
  SECTION("same explicit value: accepted") {
    pot.add("EMBF0", "FREE");
    task.add("EMBF0", "FREE");
    REQUIRE_NOTHROW(check_embf0_precedence(pot, task, "pot.tadah"));
  }
  SECTION("different explicit values: refused") {
    pot.add("EMBF0", "FREE");
    task.add("EMBF0", "ZERO");
    REQUIRE_THROWS_WITH(check_embf0_precedence(pot, task, "pot.tadah"),
                        Catch::Contains("EMBF0 conflict"));
  }
  SECTION("potential without the key: the task value applies") {
    task.add("EMBF0", "FREE");
    REQUIRE_NOTHROW(check_embf0_precedence(pot, task, "pot.tadah"));
    Context merged = Context::merge(pot, task);
    REQUIRE(merged.stored_value("EMBF0") == std::optional<std::string>("FREE"));
  }
  SECTION("no DM_REAM block: ignored") {
    Context pair = make_pair_context();
    task.add("EMBF0", "FREE");
    REQUIRE_NOTHROW(check_embf0_precedence(pair, task, "pair.tadah"));
  }
}

TEST_CASE("EMBF0: least-squares training under ZERO",
          "[embf0][train]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;

  // Reference data: a FREE DM_REAM "truth" with F(0) != 0 plus a per-atom
  // constant, i.e. absolute energies the ZERO model cannot represent without
  // an intercept.
  const std::size_t K = n_weights(make_ream_context(), ws);
  std::vector<double> w_ref(K);
  for (std::size_t k = 0; k < K; ++k) w_ref[k] = 0.7 - 0.23 * k;
  Context ref_ctx = make_ream_context();
  pin_all(ref_ctx, w_ref);
  ref_ctx.add("EMBF0", "FREE");
  Trainer ref(ref_ctx, ws);
  ref.provision_fixed_weights();
  const double e0 = -1.3;

  std::mt19937 rng(20260925);
  StructureDB db;
  const std::size_t Ntrain = 40;
  db.dbidx = {0, Ntrain};
  for (std::size_t i = 0; i < Ntrain; ++i) {
    Structure st = cluster(rng, 2 + i % 6);
    st.energy = energy_at(ref, st) + e0 * static_cast<double>(st.natoms());
    db.structures.push_back(st);
  }
  std::vector<Structure> test;
  for (std::size_t i = 0; i < 6; ++i) test.push_back(cluster(rng, 3 + i));

  SECTION("BIAS true: the ZERO fit equals the FREE fit") {
    Context cz = make_ream_context(/*bias=*/true);
    Context cf = make_ream_context(/*bias=*/true);
    cf.add("EMBF0", "FREE");
    Trainer tz(cz, ws), tf(cf, ws);
    tz.train(db);
    tf.train(db);
    for (const auto &st : test) {
      const double ez = energy_at(tz, st), ef = energy_at(tf, st);
      REQUIRE(ez == Approx(ef).epsilon(1e-7).margin(1e-9));
    }
    const double ez_iso = energy_at(tz, isolated_atom());
    const double ef_iso = energy_at(tf, isolated_atom());
    REQUIRE(ez_iso == Approx(ef_iso).epsilon(1e-7).margin(1e-9));
  }
  SECTION("BIAS false (FullLS): the isolated atom is exactly 0") {
    Context cz = make_ream_context();
    Trainer tz(cz, ws);
    tz.train(db);
    REQUIRE(energy_at(tz, isolated_atom()) == 0.0);
  }
  SECTION("PartialLS and AllFixedData: the isolated atom is exactly 0") {
    Context cp = make_ream_context();
    cp.add("FIXINDEX", "1,2");
    cp.add("FIXWEIGHT", 0.1);
    cp.add("FIXWEIGHT", -0.2);
    Trainer tp(cp, ws);
    tp.train(db);
    REQUIRE(energy_at(tp, isolated_atom()) == 0.0);

    Context ca = make_ream_context();
    pin_all(ca, w_ref);
    Trainer ta(ca, ws);
    ta.train(db);
    REQUIRE(energy_at(ta, isolated_atom()) == 0.0);
    // Same weights under FREE: the isolated atom is the raw F(0).
    Context cf = make_ream_context();
    pin_all(cf, w_ref);
    cf.add("EMBF0", "FREE");
    Trainer tf(cf, ws);
    tf.train(db);
    const double f0 =
        *tf.DCS.dmb->raw_embedding_f0(tf.model->get_weights());
    REQUIRE(f0 != 0.0);
    REQUIRE(energy_at(tf, isolated_atom()) == Approx(f0).epsilon(1e-14));
  }
}
