// test_load_transforms_weights.cpp
//
// The transforms that prepare the training data (apply_load_transforms)
// and the weights of the data.
//
//  - A structure with eweight 0 has no energy of its own (its energy line,
//    which the dataset format needs, holds a placeholder): the energy
//    shift derived from the data (ESHIFT_ATOM, the isolated atoms;
//    ESHIFT_DBATOM, the least-squares fit), the energy filter (EFILTER)
//    and the reference state of the log leave it out. A structure with
//    fweight 0 has no forces of its own: the force filter (FFILTER) leaves
//    it out.
//  - A species that only structures with eweight 0 contain has no energy
//    to fit: ESHIFT_DBATOM fits the species the energies determine and
//    gives that one 0, with a WARNING naming it.
//  - WDBFILE, the priority of each dataset, applies in the fit
//    (row_weights.h): no transform changes a priority of the data, and the
//    fit refuses a WDBFILE value that is not a finite number >= 0 (naming
//    the dataset) and a WDBFILE without one value per dataset (for its
//    size, before its signs are read). WDBFILE_AUTO and EWEIGHT_TEMP are
//    retired: Context::add refuses them, whatever their value.

#include "catch2/catch.hpp"
#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/row_weights.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <tadah/core/logger.h>

#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
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

/* What the fit's priorities of @p stdb (dataset_priorities) throw for
 * @p c, or "no exception". */
std::string priorities_refusal(const StructureDB &stdb, const Context &c) {
  try {
    (void)dataset_priorities(c, stdb);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

/* What @p f writes to std::cout (where the Logger prints) at WARNING
 * level; the stream and the level are restored afterwards. */
template <class F>
std::string warnings_of(F f) {
  Logger &log = Logger::instance();
  const LogLevel old_level = log.getLogLevel();
  log.setLogLevel(LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  try {
    f();
  } catch (...) {
    std::cout.rdbuf(old_buf);
    log.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  return captured.str();
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

/* Pins: a species that only a structure of eweight 0 contains (Nb2, its
 * placeholder energy 0) has no energy to fit: ESHIFT_DBATOM fits Ti to the
 * structures with an energy, Ti (-3 eV) and Ti2 (-6 eV), exactly (-3 to
 * 1e-10), gives Nb 0 and names it in a WARNING with its structure of
 * eweight 0. Without the structure of eweight 0 (no Nb at all) no WARNING
 * is written.
 * Before the fix: the normal matrix had a zero row for Nb, so it was
 * singular and every ESHIFT was set to 0, Ti's too ('normal-equations
 * matrix is singular; dataset compositions may be linearly dependent'). */
TEST_CASE("ESHIFT_DBATOM fits the species with an energy and names one "
          "without",
          "[load_transforms][weight0]") {
  StructureDB stdb;
  stdb.add(make_structure("Ti", -3.0, {{kTi, Vec3d(0, 0, 0)}}));
  stdb.add(make_structure("Ti2", -6.0, {{kTi, Vec3d(0, 0, 0)}, {kTi, Vec3d(2, 0, 0)}}));
  stdb.add(make_structure("Nb2 no energy", 0.0,
                          {{kNb, Vec3d(0, 0, 0)}, {kNb, Vec3d(2, 0, 0)}}, 0.0));
  stdb.dbidx = {0, 3};
  std::map<int, double> shift;
  const std::string log =
      warnings_of([&] { shift = derive_eshift_by_lsq_fit(stdb); });
  INFO(log);
  CHECK(shift[Z(kTi)] == Approx(-3.0).epsilon(1e-10));
  REQUIRE(shift.count(Z(kNb)) == 1);
  CHECK(shift[Z(kNb)] == 0.0);
  CHECK_THAT(log, Contains("WARNING") && Contains("ESHIFT_DBATOM") &&
                      Contains("Nb") && Contains("eweight 0") &&
                      !Contains("singular"));

  StructureDB ti;
  ti.add(make_structure("Ti", -3.0, {{kTi, Vec3d(0, 0, 0)}}));
  ti.add(make_structure("Ti2", -6.0, {{kTi, Vec3d(0, 0, 0)}, {kTi, Vec3d(2, 0, 0)}}));
  ti.dbidx = {0, 2};
  const std::string quiet = warnings_of([&] { shift = derive_eshift_by_lsq_fit(ti); });
  CHECK(shift[Z(kTi)] == Approx(-3.0).epsilon(1e-10));
  CHECK(quiet.find("ESHIFT_DBATOM") == std::string::npos);
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

/* Pins: a negative WDBFILE value is refused by the fit's priorities
 * (dataset_priorities) naming WDBFILE, the value and the dataset (2 of 2);
 * WDBFILE 0 (a dataset out of the fit) is accepted, and the load transforms
 * leave the priorities of the data as they are (WDBFILE applies in the fit
 * only).
 * Before the fix: -1 was applied (the weights of dataset 2 became -1);
 * after it was refused, apply_load_transforms refused it and multiplied the
 * weights by WDBFILE 0 (eweight 0). */
TEST_CASE("A negative WDBFILE value is refused", "[load_transforms][refusal]") {
  Context c;
  c.add("WDBFILE", "1 -1");
  CHECK_THAT(priorities_refusal(two_datasets(), c),
             Contains("WDBFILE") && Contains("-1") && Contains("dataset 2 of 2") &&
                 Contains(">= 0"));
  CHECK(transform_refusal(two_datasets(), c) == "no exception");
  Context z;
  z.add("WDBFILE", "1 0");
  CHECK(priorities_refusal(two_datasets(), z) == "no exception");
  StructureDB stdb = two_datasets();
  REQUIRE_NOTHROW(apply_load_transforms(stdb, z));
  CHECK(stdb(1).eweight == 1.0);
}

/* Pins: WDBFILE with more values than datasets is refused for its size
 * whatever the signs of the values past the datasets: 'WDBFILE 1 -1' with
 * one dataset names the size (2 values for 1 dataset), as 'WDBFILE 1 2'
 * does, and no dataset 2.
 * Before the fix: 'WDBFILE 1 -1: the weight of dataset 2 of 1, -1, is
 * negative', a dataset that does not exist. */
TEST_CASE("A WDBFILE of more values than datasets is refused for its size",
          "[load_transforms][refusal]") {
  StructureDB one;
  one.add(make_structure("a", -3.0, {{kTi, Vec3d(0, 0, 0)}}));
  one.dbidx = {0, 1};
  for (const std::string v : {"1 -1", "1 2"}) {
    INFO("WDBFILE " << v);
    Context c;
    c.add("WDBFILE", v);
    const std::string msg = priorities_refusal(one, c);
    CHECK_THAT(msg, Contains("2 values for 1 dataset") &&
                        !Contains("dataset 2"));
  }
}

/* Pins: WDBFILE_AUTO and EWEIGHT_TEMP are retired: Context::add refuses
 * them, whatever the value (-0.5, 0.5, 1.5; 0, -300, 300), with the
 * conversion (n_d^-a; the priorities of the data).
 * Before the fix: WDBFILE_AUTO -0.5, EWEIGHT_TEMP 0 and -300 were ignored
 * (no exception), then refused by the load transforms; 0.5 and 300
 * reweighted the data. */
TEST_CASE("WDBFILE_AUTO and EWEIGHT_TEMP are refused, whatever the value",
          "[load_transforms][refusal]") {
  for (const std::string v : {"-0.5", "0.5", "1.5"}) {
    INFO("WDBFILE_AUTO " << v);
    Context c;
    REQUIRE_THROWS_WITH(c.add("WDBFILE_AUTO", v),
                        Contains("WDBFILE_AUTO") && Contains("retired") &&
                            Contains("n_d^-a"));
  }
  for (const std::string v : {"0", "-300", "300"}) {
    INFO("EWEIGHT_TEMP " << v);
    Context c;
    REQUIRE_THROWS_WITH(c.add("EWEIGHT_TEMP", v),
                        Contains("EWEIGHT_TEMP") && Contains("retired") &&
                            Contains("WDBFILE"));
  }
}
