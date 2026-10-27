// Messages of MLIP's structures, cells, transformations, neighbour cache
// and datasets: each names what it is about (the structure, the cell, the
// supercell), never the class or the function of the code, and an internal
// error says so.
#include "catch2/catch.hpp"

#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/nn_cache.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_transformations.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace tadah::mlip;

namespace {

/* The message @p f throws, or "no exception". */
std::string thrown(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

/* Two Ta atoms in the cell of rows @p a, @p b, @p c, labelled @p label. */
Structure two_ta(const std::string &label, double a[3], double b[3],
                 double c[3]) {
  tadah::core::PeriodicTable::initialize();
  Structure st;
  st.label = label;
  st.cell.load(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  const auto ta = tadah::core::PeriodicTable::find_by_symbol("Ta");
  st.add_atom(Atom(ta, 0.0, 0.0, 0.0));
  st.add_atom(Atom(ta, 1.5, 1.5, 1.5));
  return st;
}

}  // namespace

/* Pins: the refusals of a comparison of structures (Structure::is_the_same,
 * which tadah data dedup runs) name the structure, not the function: "structure
 * 'flat' (2 atoms) has a singular cell (zero, non-finite or coplanar lattice
 * vectors); its positions cannot be compared through the cell", "the
 * threshold 2.9 A is too large for structure 'cube' ..." and "the threshold
 * must be a finite number of at least 0".
 * Before the fix: each began "Structure::is_the_same: ". */
TEST_CASE("A comparison of structures names the structure, not the function",
          "[messages][is_the_same]") {
  double x[3] = {10, 0, 0}, y[3] = {0, 10, 0}, z[3] = {0, 0, 10},
         flat[3] = {10, 10, 0};
  const Structure cube = two_ta("cube", x, y, z);
  const Structure planar = two_ta("flat", x, y, flat);
  CHECK(thrown([&] { (void)planar.is_the_same(cube, 1e-6); }) ==
        "structure 'flat' (2 atoms) has a singular cell (zero, non-finite or "
        "coplanar lattice vectors); its positions cannot be compared through "
        "the cell");
  CHECK(thrown([&] { (void)cube.is_the_same(cube, 2.9); })
            .rfind("the threshold 2.9 A is too large for structure 'cube': ",
                   0) == 0);
  CHECK(thrown([&] { (void)cube.is_the_same(cube, -1.0); }) ==
        "the threshold must be a finite number of at least 0");
}

/* Pins: the cells of the PC_OBSERVABLE building blocks and the supercell
 * refuse their arguments naming what they build, with the value as given:
 * "a bcc cell needs a0 > 0, got -1.5", "an isolated-atom cell needs a box
 * size > 0, got 0", "a supercell needs every factor >= 1, got 0 1 1"; the
 * cells this version does not build are internal errors ("internal error:
 * an fcc cell is not built by this version", "... an hcp cell ...").
 * Before the fix: "cells::bcc: a0 must be > 0", "cells::isolated_atom:
 * box_size must be > 0", "StructureTransformation::replicate: every factor
 * must be >= 1, got 0 1 1." and "cells::fcc: not implemented in MVP. First
 * observable that needs FCC (e.g. phase_transition_energy) must land its
 * consumer before this stub may be filled in.". */
TEST_CASE("The cells and the supercell name what they build",
          "[messages][cell_builder]") {
  tadah::core::PeriodicTable::initialize();
  CHECK(thrown([] { (void)cells::bcc("Ta", -1.5); }) ==
        "a bcc cell needs a0 > 0, got -1.5");
  CHECK(thrown([] { (void)cells::isolated_atom("Ta", 0.0); }) ==
        "an isolated-atom cell needs a box size > 0, got 0");
  CHECK(thrown([] { (void)cells::fcc("Ta", 4.0); }) ==
        "internal error: an fcc cell is not built by this version");
  CHECK(thrown([] { (void)cells::hcp("Ta", 3.0, 1.6); }) ==
        "internal error: an hcp cell is not built by this version");
  Structure st = cells::bcc("Ta", 3.3);
  CHECK(thrown([&] { StructureTransformation::replicate(st, 0, 1, 1); }) ==
        "a supercell needs every factor >= 1, got 0 1 1");
}

/* Pins: a mirror plane (h k l) whose normal is too short to normalise (a
 * cell of 1e15 A, plane (1 0 0)) is refused naming the plane: "the plane (1
 * 0 0) of this cell has a normal of length near 0".
 * Before the fix: "Plane normal is near zero. Check cell or (h,k,l).", an
 * instruction. */
TEST_CASE("A mirror plane without a normal is named",
          "[messages][transform]") {
  double x[3] = {1e15, 0, 0}, y[3] = {0, 1e15, 0}, z[3] = {0, 0, 1e15};
  Structure huge = two_ta("huge", x, y, z);
  CHECK(thrown([&] { StructureTransformation::mirrorHkl(huge, 1, 0, 0); }) ==
        "the plane (1 0 0) of this cell has a normal of length near 0");
}

/* Pins: a cached neighbour list whose cutoff does not exceed its skin is an
 * internal error naming both as given: "internal error: the cached
 * neighbour list was built with cutoff 1.5 A, which does not exceed the
 * skin 2 A; a cached list is built at the cutoff plus the skin".
 * Before the fix: "nn_cache::is_fresh: the cached neighbour list was built
 * with cutoff 1.500000 A, ...; build it with NNFinder at cutoff = rcut +
 * skin". */
TEST_CASE("A cached neighbour list within its skin is an internal error",
          "[messages][nn_cache]") {
  double x[3] = {3.3, 0, 0}, y[3] = {0, 3.3, 0}, z[3] = {0, 0, 3.3};
  Structure st = two_ta("bcc", x, y, z);
  nn_cache::State cache;
  cache.skin = 2.0;
  NNFinder nnf;
  nnf.calc(st, 1.5);
  nn_cache::capture(st, cache);
  CHECK(thrown([&] { (void)nn_cache::is_fresh(st, cache); }) ==
        "internal error: the cached neighbour list was built with cutoff 1.5 "
        "A, which does not exceed the skin 2 A; a cached list is built at "
        "the cutoff plus the skin");
}

/* Pins: a dataset file that does not exist is refused with "DBFILE does not
 * exist: nosuch.tadah", and no log line names the function that found it.
 * Before the fix: an INFO line "StructureDB::find_unique_elements: DBFILE
 * does not exist: nosuch.tadah" came before the refusal. */
TEST_CASE("A dataset file that does not exist is named once",
          "[messages][structure_db]") {
  auto &logger = tadah::core::Logger::instance();
  const auto level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream out;
  std::streambuf *old = std::cout.rdbuf(out.rdbuf());
  const std::string what =
      thrown([] { (void)StructureDB::find_unique_elements("nosuch.tadah"); });
  std::cout.rdbuf(old);
  logger.setLogLevel(level);
  INFO("log:\n" << out.str());
  CHECK(what == "DBFILE does not exist: nosuch.tadah");
  CHECK(out.str().find("StructureDB::") == std::string::npos);
}
