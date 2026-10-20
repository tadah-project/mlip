// test_load_transforms_weights.cpp
//
// The transforms that prepare the training data (apply_load_transforms)
// and the weights of the data.
//
//  - A structure with eweight 0 has no energy of its own (tadah data
//    convert stores a placeholder for a file that gives none): the energy
//    shift derived from the data (ESHIFT_ATOM, the isolated atoms;
//    ESHIFT_DBATOM, the least-squares fit), the energy filter (EFILTER),
//    the Boltzmann weights (EWEIGHT_TEMP, its minimum energy) and the
//    reference state of the log leave it out. A structure with fweight 0
//    has no forces of its own: the force filter (FFILTER) leaves it out.
//  - The values a transform cannot apply are refused: a negative WDBFILE
//    value (naming the dataset), WDBFILE_AUTO outside [0, 1] and
//    EWEIGHT_TEMP <= 0. (WDBFILE_AUTO below 0 and EWEIGHT_TEMP <= 0 were
//    ignored, although the documentation says they are rejected; a
//    negative WDBFILE was applied.)

#include "catch2/catch.hpp"
#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace tadah::core;
using namespace tadah::mlip;
using Catch::Matchers::Contains;

namespace {

Structure make_structure(const char *label, double E,
                         std::vector<std::pair<std::string, Vec3d>> atoms,
                         double eweight = 1.0) {
  Structure st;
  st.label = label;
  st.energy = E;
  st.eweight = eweight;
  st.cell.set_zero();
  st.cell(0, 0) = st.cell(1, 1) = st.cell(2, 2) = 5.0;
  st.stress.set_zero();
  for (const auto &p : atoms) {
    Atom a_obj(PeriodicTable::find_by_symbol(p.first), p.second[0],
               p.second[1], p.second[2]);
    st.add_atom(a_obj);
  }
  return st;
}

const char *kTi = "Ti";
const char *kNb = "Nb";

int Z(const char *s) { return PeriodicTable::find_by_symbol(s).Z; }

/* What apply_load_transforms throws for @p c, or "no exception". */
std::string transform_refusal(StructureDB stdb, const Context &c) {
  try {
    apply_load_transforms(stdb, c);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

StructureDB two_datasets() {
  StructureDB stdb;
  stdb.add(make_structure("a", -3.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("b", -6.1, {{kTi, Vec3d(0, 0, 0)}, {kTi, Vec3d(2, 0, 0)}}));
  stdb.add(make_structure("c", -9.2, {{kTi, Vec3d(0, 0, 0)}, {kTi, Vec3d(2, 0, 0)},
                                      {kTi, Vec3d(0, 2, 0)}}));
  stdb.dbidx = {0, 1, 3};
  return stdb;
}

}  // namespace

/* Pins: ESHIFT_ATOM takes the isolated Ti atom with an energy (-3) and
 * leaves out the one with eweight 0 (a placeholder energy 0): the shift is
 * -3, not their mean -1.5. With only an isolated atom of eweight 0, Ti has
 * no isolated atom with an energy: its shift is 0 (and a WARNING).
 * Before the fix: -1.5, and 0 in the second case. */
TEST_CASE("ESHIFT_ATOM leaves out isolated atoms without an energy",
          "[load_transforms][weight0]") {
  StructureDB stdb;
  stdb.add(make_structure("Ti1", -3.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("Ti1 no energy", 0.0, {{kTi, Vec3d(0, 0, 0)}}, 0.0));
  stdb.add(make_structure("Ti2", -7.0, {{kTi, Vec3d(0, 0, 0)}, {kTi, Vec3d(2, 0, 0)}}));
  stdb.dbidx = {0, 3};
  CHECK(derive_eshift_from_isolated_atoms(stdb)[Z(kTi)] == Approx(-3.0));

  StructureDB only;
  only.add(make_structure("Ti1 no energy", -100.0, {{kTi, Vec3d(0, 0, 0)}}, 0.0));
  only.add(make_structure("Ti2", -7.0, {{kTi, Vec3d(0, 0, 0)}, {kTi, Vec3d(2, 0, 0)}}));
  only.dbidx = {0, 2};
  CHECK(derive_eshift_from_isolated_atoms(only)[Z(kTi)] == 0.0);
}

/* Pins: ESHIFT_DBATOM fits E = N_Ti a + N_Nb b to the structures with an
 * energy only: with exact energies for a = -3, b = -7 and a structure of
 * eweight 0 whose placeholder energy is 0, the fit is exact (a = -3,
 * b = -7 to 1e-10).
 * Before the fix: the placeholder pulled both away. */
TEST_CASE("ESHIFT_DBATOM leaves out structures without an energy",
          "[load_transforms][weight0]") {
  StructureDB stdb;
  stdb.add(make_structure("Ti", -3.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("TiNb", -10.0, {{kTi, Vec3d(0, 0, 0)}, {kNb, Vec3d(2, 0, 0)}}));
  stdb.add(make_structure("Nb2", -14.0, {{kNb, Vec3d(0, 0, 0)}, {kNb, Vec3d(2, 0, 0)}}));
  stdb.add(make_structure("TiNb2 no energy", 0.0,
                          {{kTi, Vec3d(0, 0, 0)}, {kNb, Vec3d(2, 0, 0)},
                           {kNb, Vec3d(0, 2, 0)}},
                          0.0));
  stdb.dbidx = {0, 4};
  auto shift = derive_eshift_by_lsq_fit(stdb);
  CHECK(shift[Z(kTi)] == Approx(-3.0).epsilon(1e-10));
  CHECK(shift[Z(kNb)] == Approx(-7.0).epsilon(1e-10));
}

/* Pins: EFILTER -10 0 keeps a structure of eweight 0 whose placeholder
 * energy per atom (+5) is outside the window (its energy is not data) and
 * drops one of eweight 1 at +3; FFILTER 50 keeps a structure of fweight 0
 * whose placeholder force (100) is above it and drops one of fweight 1.
 * Before the fix: both placeholders were dropped (4 dropped, not 2). */
TEST_CASE("EFILTER and FFILTER leave out the energies and forces a structure "
          "does not have",
          "[load_transforms][weight0]") {
  StructureDB stdb;
  stdb.add(make_structure("normal", -2.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("hot", 3.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("no energy", 5.0, {{kTi, Vec3d(0, 0, 0)}}, 0.0));
  Structure bad = make_structure("badF", -2.0, {{kTi, Vec3d(0, 0, 0)}});
  bad.atoms[0].force = Vec3d(100.0, 0.0, 0.0);
  stdb.add(bad);
  Structure nof = make_structure("no forces", -2.0, {{kTi, Vec3d(0, 0, 0)}});
  nof.atoms[0].force = Vec3d(100.0, 0.0, 0.0);
  nof.fweight = 0.0;
  stdb.add(nof);
  stdb.dbidx = {0, stdb.size()};
  OutlierCounts c = apply_outlier_filter(stdb, -10.0, 0.0, 50.0);
  CHECK(c.energy == 1);
  CHECK(c.force == 1);
  REQUIRE(stdb.size() == 3);
  CHECK(stdb(0).label == "normal");
  CHECK(stdb(1).label == "no energy");
  CHECK(stdb(2).label == "no forces");
}

/* Pins: EWEIGHT_TEMP's minimum energy is that of the structures with an
 * energy: with a structure of eweight 0 at a placeholder -100 eV/atom, the
 * structure at 0 eV/atom keeps eweight 1 (it is the minimum) and the
 * placeholder's eweight stays 0 (no nan).
 * Before the fix: the minimum was the placeholder's, so the structure at 0
 * got exp(-100/kT), 0 at 1000 K. */
TEST_CASE("EWEIGHT_TEMP takes its minimum energy from the structures with an "
          "energy",
          "[load_transforms][weight0]") {
  StructureDB stdb;
  stdb.add(make_structure("min", 0.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("hot", 0.01, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("no energy", -100.0, {{kTi, Vec3d(0, 0, 0)}}, 0.0));
  stdb.dbidx = {0, 3};
  apply_boltzmann_eweight(stdb, 1000.0);
  const double kB = 8.617333262e-5;
  CHECK(stdb(0).eweight == Approx(1.0));
  CHECK(stdb(1).eweight == Approx(std::exp(-0.01 / (kB * 1000.0))));
  CHECK(stdb(2).eweight == 0.0);
}

/* Pins: a negative WDBFILE value is refused naming WDBFILE, the value and
 * the dataset (2 of 2); WDBFILE 0 (a dataset out of the fit) is applied.
 * Before the fix: -1 was applied (the weights of dataset 2 became -1). */
TEST_CASE("A negative WDBFILE value is refused", "[load_transforms][refusal]") {
  Context c;
  c.add("WDBFILE", "1 -1");
  CHECK_THAT(transform_refusal(two_datasets(), c),
             Contains("WDBFILE") && Contains("-1") && Contains("dataset 2 of 2") &&
                 Contains(">= 0"));
  Context z;
  z.add("WDBFILE", "1 0");
  StructureDB stdb = two_datasets();
  REQUIRE_NOTHROW(apply_load_transforms(stdb, z));
  CHECK(stdb(1).eweight == 0.0);
}

/* Pins: WDBFILE_AUTO -0.5 and 1.5 are refused naming the key, the value
 * and [0, 1]; EWEIGHT_TEMP 0 and -300 are refused naming the key, the value
 * and > 0, as training.rst says they are.
 * Before the fix: WDBFILE_AUTO -0.5, EWEIGHT_TEMP 0 and -300 were ignored
 * (no exception); WDBFILE_AUTO 1.5 was refused. */
TEST_CASE("WDBFILE_AUTO outside [0, 1] and EWEIGHT_TEMP <= 0 are refused",
          "[load_transforms][refusal]") {
  for (const std::string v : {"-0.5", "1.5"}) {
    INFO("WDBFILE_AUTO " << v);
    Context c;
    c.add("WDBFILE_AUTO", v);
    CHECK_THAT(transform_refusal(two_datasets(), c),
               Contains("WDBFILE_AUTO") && Contains(v) && Contains("[0, 1]"));
  }
  for (const std::string v : {"0", "-300"}) {
    INFO("EWEIGHT_TEMP " << v);
    Context c;
    c.add("EWEIGHT_TEMP", v);
    CHECK_THAT(transform_refusal(two_datasets(), c),
               Contains("EWEIGHT_TEMP") && Contains(v) && Contains("> 0"));
  }
}
