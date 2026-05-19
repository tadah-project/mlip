// test_load_transforms.cpp
#include "catch2/catch.hpp"
#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <cmath>

using namespace tadah::core;
using namespace tadah::mlip;

namespace {

Structure make_structure(const char *label, double E,
                         std::vector<std::pair<std::string, Vec3d>> atoms,
                         double a = 5.0) {
  Structure st;
  st.label = label;
  st.energy = E;
  st.cell.set_zero();
  st.cell(0, 0) = a;
  st.cell(1, 1) = a;
  st.cell(2, 2) = a;
  st.stress.set_zero();
  for (const auto &p : atoms) {
    Atom a_obj(PeriodicTable::find_by_symbol(p.first), p.second[0],
               p.second[1], p.second[2]);
    st.add_atom(a_obj);
  }
  return st;
}

}  // namespace

TEST_CASE("apply_eshift subtracts per-element baseline", "[load_transforms]") {
  StructureDB stdb;
  // Single Ti atom config: E = -3.0
  stdb.add(make_structure("Ti1", -3.0, {{"Ti", Vec3d(0, 0, 0)}}));
  // Four-Ti config: E = -11.0
  stdb.add(make_structure(
      "Ti4", -11.0,
      {{"Ti", Vec3d(0, 0, 0)},
       {"Ti", Vec3d(1, 0, 0)},
       {"Ti", Vec3d(0, 1, 0)},
       {"Ti", Vec3d(0, 0, 1)}}));
  stdb.dbidx = {0, 2};

  std::map<int, double> shift{{PeriodicTable::find_by_symbol("Ti").Z, -3.0}};
  apply_eshift(stdb, shift);

  // Single-atom: -3.0 - (-3.0) = 0
  REQUIRE(stdb(0).energy == Approx(0.0));
  // Four-atom: -11.0 - 4*(-3.0) = -11.0 + 12.0 = +1.0
  REQUIRE(stdb(1).energy == Approx(1.0));
}

TEST_CASE("derive_eshift_from_isolated_atoms recovers the per-Z energy",
          "[load_transforms]") {
  StructureDB stdb;
  stdb.add(make_structure("Ti1", -3.0, {{"Ti", Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("Nb1", -7.0, {{"Nb", Vec3d(0, 0, 0)}}));
  stdb.add(make_structure(
      "TiNb", -11.0,
      {{"Ti", Vec3d(0, 0, 0)}, {"Nb", Vec3d(1, 0, 0)}}));
  stdb.dbidx = {0, 3};

  auto eshift = derive_eshift_from_isolated_atoms(stdb);
  REQUIRE(eshift[PeriodicTable::find_by_symbol("Ti").Z] == Approx(-3.0));
  REQUIRE(eshift[PeriodicTable::find_by_symbol("Nb").Z] == Approx(-7.0));
}

TEST_CASE("derive_eshift_by_lsq_fit fits exact composition energies",
          "[load_transforms]") {
  // Build configs whose energies are exactly E = N_Ti * a + N_Nb * b for
  // a = -3.0, b = -7.0. The fit should recover a and b.
  StructureDB stdb;
  const double a = -3.0;
  const double b = -7.0;

  auto e = [&](int nTi, int nNb) { return nTi * a + nNb * b; };
  std::vector<std::pair<int, int>> compositions{
      {1, 0}, {0, 1}, {2, 0}, {0, 2}, {1, 1}, {3, 1}, {2, 2}, {1, 2}};
  for (auto p : compositions) {
    std::vector<std::pair<std::string, Vec3d>> atoms;
    for (int i = 0; i < p.first; ++i)
      atoms.emplace_back("Ti", Vec3d(i * 1.0, 0, 0));
    for (int i = 0; i < p.second; ++i)
      atoms.emplace_back("Nb", Vec3d(0, i * 1.0, 0));
    stdb.add(make_structure("c", e(p.first, p.second), atoms));
  }
  stdb.dbidx = {0, stdb.size()};

  auto eshift = derive_eshift_by_lsq_fit(stdb);
  REQUIRE(eshift[PeriodicTable::find_by_symbol("Ti").Z] == Approx(a).epsilon(1e-10));
  REQUIRE(eshift[PeriodicTable::find_by_symbol("Nb").Z] == Approx(b).epsilon(1e-10));
}

TEST_CASE("apply_outlier_filter drops only structures violating thresholds",
          "[load_transforms]") {
  StructureDB stdb;
  // E/N: -2 / 1 = -2, -8 / 4 = -2, +3 / 1 = +3, max-force 0.5 vs 100
  stdb.add(make_structure("normal_a", -2.0, {{"Ti", Vec3d(0, 0, 0)}}));
  stdb.add(make_structure(
      "normal_b", -8.0,
      {{"Ti", Vec3d(0, 0, 0)},
       {"Ti", Vec3d(1, 0, 0)},
       {"Ti", Vec3d(0, 1, 0)},
       {"Ti", Vec3d(0, 0, 1)}}));
  stdb.add(make_structure("hot", +3.0, {{"Ti", Vec3d(0, 0, 0)}}));

  Structure bad_force = make_structure("badF", -2.0, {{"Ti", Vec3d(0, 0, 0)}});
  bad_force.atoms[0].force = Vec3d(100.0, 0.0, 0.0);
  stdb.add(bad_force);

  stdb.dbidx = {0, stdb.size()};

  OutlierCounts c =
      apply_outlier_filter(stdb, /*emin*/ -10.0, /*emax*/ 0.0, /*fmax*/ 50.0);
  REQUIRE(c.energy == 1);  // "hot"
  REQUIRE(c.force == 1);   // "badF"
  REQUIRE(stdb.size() == 2);
  REQUIRE(stdb(0).label == std::string("normal_a"));
  REQUIRE(stdb(1).label == std::string("normal_b"));
}

TEST_CASE("auto_balance_db_weights computes N_i^(-alpha)", "[load_transforms]") {
  StructureDB stdb;
  for (int i = 0; i < 2; ++i)
    stdb.add(make_structure("ds0", -1.0, {{"Ti", Vec3d(0, 0, 0)}}));
  for (int i = 0; i < 1000; ++i)
    stdb.add(make_structure("ds1", -2.0, {{"Ti", Vec3d(0, 0, 0)}}));
  stdb.dbidx = {0, 2, 1002};

  auto w0 = auto_balance_db_weights(stdb, 0.0);
  REQUIRE(w0.size() == 2);
  REQUIRE(w0[0] == Approx(1.0));
  REQUIRE(w0[1] == Approx(1.0));

  auto w_half = auto_balance_db_weights(stdb, 0.5);
  REQUIRE(w_half[0] == Approx(1.0 / std::sqrt(2.0)));
  REQUIRE(w_half[1] == Approx(1.0 / std::sqrt(1000.0)));

  auto w_full = auto_balance_db_weights(stdb, 1.0);
  REQUIRE(w_full[0] == Approx(0.5));
  REQUIRE(w_full[1] == Approx(0.001));

  REQUIRE_THROWS_AS(auto_balance_db_weights(stdb, -0.1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(auto_balance_db_weights(stdb, 1.1),
                    std::invalid_argument);
}

TEST_CASE("apply_per_db_weights composes with WDBFILE_AUTO", "[load_transforms]") {
  StructureDB stdb;
  for (int i = 0; i < 2; ++i)
    stdb.add(make_structure("ds0", -1.0, {{"Ti", Vec3d(0, 0, 0)}}));
  for (int i = 0; i < 4; ++i)
    stdb.add(make_structure("ds1", -2.0, {{"Ti", Vec3d(0, 0, 0)}}));
  stdb.dbidx = {0, 2, 6};

  auto w_auto = auto_balance_db_weights(stdb, 0.5);
  std::vector<double> w_user{5.0, 1.0};
  std::vector<double> w_final(2);
  for (int i = 0; i < 2; ++i) w_final[i] = w_user[i] * w_auto[i];

  apply_per_db_weights(stdb, w_final);
  REQUIRE(stdb(0).eweight == Approx(5.0 / std::sqrt(2.0)));
  REQUIRE(stdb(2).eweight == Approx(1.0 / std::sqrt(4.0)));
  REQUIRE(stdb(2).fweight == Approx(1.0 / std::sqrt(4.0)));
  REQUIRE(stdb(2).sweight == Approx(1.0 / std::sqrt(4.0)));
}

TEST_CASE("apply_boltzmann_eweight rewweights toward minimum energy",
          "[load_transforms]") {
  StructureDB stdb;
  // Two configs, same N=1, energies 0.0 (E_min) and 0.01 eV/atom.
  stdb.add(make_structure("min", 0.0, {{"Ti", Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("hot", 0.01, {{"Ti", Vec3d(0, 0, 0)}}));
  stdb.dbidx = {0, 2};

  apply_boltzmann_eweight(stdb, 1000.0);  // kT ~ 0.0862 eV
  REQUIRE(stdb(0).eweight == Approx(1.0));  // exp(0) = 1
  // The relative weight should be exp(-0.01/(kB*1000)) ~= exp(-0.116)
  const double kB = 8.617333262e-5;
  REQUIRE(stdb(1).eweight == Approx(std::exp(-0.01 / (kB * 1000.0))));
}

TEST_CASE("apply_com_force_zero zeroes net force per config",
          "[load_transforms]") {
  StructureDB stdb;
  Structure st = make_structure(
      "x", -1.0, {{"Ti", Vec3d(0, 0, 0)}, {"Ti", Vec3d(1, 0, 0)}});
  st.atoms[0].force = Vec3d(2.0, 0.0, 0.0);
  st.atoms[1].force = Vec3d(0.0, 0.0, 0.0);
  stdb.add(st);
  stdb.dbidx = {0, 1};

  apply_com_force_zero(stdb);
  Vec3d sum = stdb(0).atoms[0].force + stdb(0).atoms[1].force;
  REQUIRE(sum.norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("stamp_into_param_file populates LSCALE / ESHIFT keys",
          "[load_transforms]") {
  // pot.tadah is a serialised Context. The LAMMPS plugin parses it via
  // CoreUserConfigParser and silently ignores LSCALE/ESHIFT. This test asserts
  // that stamp_into_param_file produces a Context with the keys present and
  // values intact, which is the contract that pot.tadah → predict and pot.tadah
  // → LAMMPS rely on.
  PeriodicTable::initialize();
  LoadTransformReport rep;
  rep.applied = true;
  rep.lscale = 1.005;
  rep.eshift[PeriodicTable::find_by_symbol("Ti").Z] = -3.0;
  rep.eshift[PeriodicTable::find_by_symbol("Nb").Z] = -7.0;
  rep.eshift_source = LoadTransformReport::EshiftSource::Atom;

  Context pot;
  stamp_into_param_file(pot, rep);
  REQUIRE(pot.contains("LSCALE"));
  REQUIRE(pot.get<double>("LSCALE") == Approx(1.005));
  REQUIRE(pot.contains("ESHIFT"));
  REQUIRE(pot.size("ESHIFT") == 2);
  // ESHIFT entries are written sorted by Z; Ti=22 < Nb=41
  REQUIRE(pot.get<double>("ESHIFT", 0) == Approx(-3.0));
  REQUIRE(pot.get<double>("ESHIFT", 1) == Approx(-7.0));
}

TEST_CASE("apply_load_transforms — round trip ESHIFT_ATOM + LSCALE",
          "[load_transforms]") {
  StructureDB stdb;
  // Single-atom Ti config (the auto baseline)
  stdb.add(make_structure("Ti1", -3.0, {{"Ti", Vec3d(0, 0, 0)}}));
  Structure four = make_structure(
      "Ti4", -11.0,
      {{"Ti", Vec3d(0, 0, 0)},
       {"Ti", Vec3d(1, 0, 0)},
       {"Ti", Vec3d(0, 1, 0)},
       {"Ti", Vec3d(0, 0, 1)}});
  four.atoms[0].force = Vec3d(0.5, 0.0, 0.0);
  four.stress(0, 0) = 0.7;
  stdb.add(four);
  stdb.dbidx = {0, 2};

  Context ctx;
  ctx.add("LSCALE", 1.05);
  ctx.add("ESHIFT_ATOM", true);

  auto report = apply_load_transforms(stdb, ctx);
  REQUIRE(report.applied == true);
  REQUIRE(report.lscale == Approx(1.05));
  REQUIRE(report.eshift_source == LoadTransformReport::EshiftSource::Atom);
  REQUIRE(report.eshift[PeriodicTable::find_by_symbol("Ti").Z] ==
          Approx(-3.0));

  // After ESHIFT_ATOM (subtract -3.0 per atom): single-atom -> 0; four-atom -> +1
  REQUIRE(stdb(0).energy == Approx(0.0));
  REQUIRE(stdb(1).energy == Approx(1.0));

  // After LSCALE 1.05: positions scaled, force divided, stress invariant
  REQUIRE(stdb(1).cell(0, 0) == Approx(5.0 * 1.05));
  REQUIRE(stdb(1).atoms[0].force[0] == Approx(0.5 / 1.05));
  REQUIRE(stdb(1).stress(0, 0) == Approx(0.7));
}
