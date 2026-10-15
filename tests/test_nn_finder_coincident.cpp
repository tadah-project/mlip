// test_nn_finder_coincident.cpp
//
// NNFinder refuses two atoms at one position. A pair whose distance,
// computed as consumers compute it (r_p - (r_q + s0 a0 + s1 a1 + s2 a2)),
// is exactly 0 throws std::invalid_argument naming the two atoms (1-based
// indices and element symbols), the position and the image shift s (in
// the stored basis, the NNTable convention: the image of the second atom
// shifted by s lies on the first); calc(StructureDB&) adds the structure
// ("structure <i> of <N> ('<label>')"). Every descriptor divides by the
// distance, so such a pair made the forces and the stress NaN: `tadah
// predict` printed nan, `tadah train` stopped in DGELSD without naming the
// structure, an energy-only fit silently fitted it, and `tadah hpo` failed
// every evaluation.
//
// Two atoms close together but apart (1e-12 A) are a data-quality question
// whose harm depends on the basis: they are listed, not refused.
//
// The geometries are built here; NNFinder is the only production code used.

#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <array>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>

using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

namespace {

constexpr double kCutoff = 5.0;

Structure with_rows(double a0x, double a0y, double a0z, double a1x, double a1y,
                    double a1z, double a2x, double a2y, double a2z) {
  Structure st;
  st.cell.load(a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z);
  return st;
}

Structure cube(double a) {
  return with_rows(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
}

void add(Structure &st, const char *symbol, double x, double y, double z) {
  st.add_atom(Atom(PeriodicTable::find_by_symbol(symbol), x, y, z, 0.0, 0.0,
                   0.0));
}

/* A 3.3 A Ta cube: atom 1 at (0, 0, 0), atom 2 at the cube centre and
 * atom 3 at (dx, 0, 0), so atoms 1 and 3 are dx apart. */
Structure ta3(double dx) {
  Structure st = cube(3.3);
  add(st, "Ta", 0.0, 0.0, 0.0);
  add(st, "Ta", 1.65, 1.65, 1.65);
  add(st, "Ta", dx, 0.0, 0.0);
  return st;
}

/* The message of the std::invalid_argument that @p f throws; "" when it
 * throws nothing, and "other exception: <what>" for any other exception. */
template <class F>
std::string invalid_argument_message(F f) {
  try {
    f();
  } catch (const std::invalid_argument &e) {
    return e.what();
  } catch (const std::exception &e) {
    return std::string("other exception: ") + e.what();
  }
  return "";
}

bool contains(const std::string &s, const std::string &part) {
  return s.find(part) != std::string::npos;
}

/* Whether atom @p i of @p st lists atom @p j (0-based) with the image
 * shift @p s. */
bool lists(const Structure &st, std::size_t i, std::size_t j,
           const std::array<double, 3> &s) {
  if (!st.nn) return false;
  for (std::size_t k = 0; k < st.nn->idx[i].size(); ++k)
    if (st.nn->idx[i][k] == j && st.nn->shift[i][k][0] == s[0] &&
        st.nn->shift[i][k][1] == s[1] && st.nn->shift[i][k][2] == s[2])
      return true;
  return false;
}

}  // namespace

/* Pins: in a 3.3 A cube with Ta at (0, 0, 0), Ta at (1.65, 1.65, 1.65) and
 * a third Ta at (0, 0, 0), NNFinder::calc and NNFinder::build (on a const
 * structure) throw std::invalid_argument whose message names atoms 1 and 3
 * with their symbols ("atoms 1 and 3 (Ta, Ta)"), says they are at the same
 * position (0, 0, 0), gives the image shift (0, 0, 0) and says why ("not a
 * physical structure"). With the third atom at (1e-12, 0, 0) instead the
 * list is built, and atoms 1 and 3 list each other with no image shift: a
 * close pair is not refused.
 * Before the fix: the pair was listed at distance 0 and nothing was thrown,
 * so both CHECK_THROWS_AS and every message CHECK failed (the message was
 * empty). */
TEST_CASE("NNFinder: two atoms at the same position are refused, naming the "
          "atoms, the position and the image shift, and a pair 1e-12 A apart "
          "is listed",
          "[nn][coincident]") {
  PeriodicTable::initialize();
  {
    Structure st = ta3(0.0);
    CHECK_THROWS_AS(NNFinder().calc(st, kCutoff), std::invalid_argument);
    const Structure &cst = st;
    CHECK_THROWS_AS(NNFinder().build(cst, kCutoff), std::invalid_argument);
    const std::string msg =
        invalid_argument_message([&] { NNFinder().calc(st, kCutoff); });
    INFO("message: " << msg);
    CHECK(contains(msg, "atoms 1 and 3 (Ta, Ta)"));
    CHECK(contains(msg, "same position (0, 0, 0)"));
    CHECK(contains(msg, "image shift (0, 0, 0)"));
    CHECK(contains(msg, "not a physical structure"));
  }
  {
    Structure st = ta3(1e-12);
    REQUIRE_NOTHROW(NNFinder().calc(st, kCutoff));
    CHECK(lists(st, 0, 2, {0.0, 0.0, 0.0}));
    CHECK(lists(st, 2, 0, {0.0, 0.0, 0.0}));
  }
}

/* Pins: an atom on a periodic image of another is refused in the same way,
 * and the image shift is given in the stored basis.
 * - bcc Fe, a = 2.8665 A, with Fe at fractional (0, 0, 0), (1, 0, 0) and
 *   (0.5, 0.5, 0.5): a CIF that lists both corners.
 *   The image of atom 2 shifted by -a0 lies on atom 1: "atoms 1 and 2
 *   (Fe, Fe)", "same position (0, 0, 0)", "image shift (-1, 0, 0)".
 * - The sheared description (4,0,0), (12,4,0), (0,0,4) of a simple cubic
 *   lattice (NNFinder reduces it to a 4 A cube before the search), Nb at
 *   (3.5, 1, 2) and Ta at that position plus a1, (15.5, 5, 2): the image of
 *   atom 2 shifted by -a1 lies on atom 1, so "atoms 1 and 2 (Nb, Ta)", "same
 *   position (3.5, 1, 2)" and "image shift (0, -1, 0)" (the stored basis,
 *   not the reduced one the search uses). Every coordinate is a dyadic
 *   number, so the distance is exactly 0 whatever the order of the
 *   additions.
 * Before the fix: atom 1 listed atom 2 at distance 0 with that shift and
 * nothing was thrown, so the CHECK_THROWS_AS and the message CHECKs
 * failed. */
TEST_CASE("NNFinder: an atom on a periodic image of another is refused, "
          "naming the image shift in the stored basis",
          "[nn][coincident]") {
  PeriodicTable::initialize();
  Structure st;
  std::string atoms, position, shift;
  SECTION("bcc Fe with atoms at fractional x = 0 and x = 1") {
    const double a = 2.8665;
    st = cube(a);
    add(st, "Fe", 0.0, 0.0, 0.0);
    add(st, "Fe", 1.0 * a, 0.0, 0.0);
    add(st, "Fe", 0.5 * a, 0.5 * a, 0.5 * a);
    atoms = "atoms 1 and 2 (Fe, Fe)";
    position = "same position (0, 0, 0)";
    shift = "image shift (-1, 0, 0)";
  }
  SECTION("a sheared description of a simple cubic lattice") {
    st = with_rows(4.0, 0.0, 0.0, 12.0, 4.0, 0.0, 0.0, 0.0, 4.0);
    add(st, "Nb", 3.5, 1.0, 2.0);
    add(st, "Ta", 15.5, 5.0, 2.0);
    atoms = "atoms 1 and 2 (Nb, Ta)";
    position = "same position (3.5, 1, 2)";
    shift = "image shift (0, -1, 0)";
  }
  CHECK_THROWS_AS(NNFinder().calc(st, kCutoff), std::invalid_argument);
  const std::string msg =
      invalid_argument_message([&] { NNFinder().calc(st, kCutoff); });
  INFO("message: " << msg);
  CHECK(contains(msg, atoms));
  CHECK(contains(msg, position));
  CHECK(contains(msg, shift));
}

/* Pins: calc on a StructureDB names the structure that holds the pair: of a
 * clean bcc Ta cell and the 3-atom cell with atoms 1 and 3 at (0, 0, 0),
 * labelled "Ta3 coincident", the refusal is a std::invalid_argument naming
 * "structure 2 of 2", "'Ta3 coincident'", "atoms 1 and 3 (Ta, Ta)" and "same
 * position". This is the message `tadah train`, `tadah predict` and `tadah
 * hpo` report.
 * Before the fix: nothing was thrown, so the CHECK_THROWS_AS and every
 * message CHECK failed. */
TEST_CASE("NNFinder: calc on a StructureDB names the structure with two atoms "
          "at the same position",
          "[nn][coincident][stdb]") {
  PeriodicTable::initialize();
  Structure clean = cube(3.3);
  add(clean, "Ta", 0.0, 0.0, 0.0);
  add(clean, "Ta", 1.65, 1.65, 1.65);
  clean.label = "clean bcc";
  Structure coincident = ta3(0.0);
  coincident.label = "Ta3 coincident";
  StructureDB db;
  db.add(clean);
  db.add(coincident);

  CHECK_THROWS_AS(NNFinder().calc(db, kCutoff), std::invalid_argument);
  const std::string msg =
      invalid_argument_message([&] { NNFinder().calc(db, kCutoff); });
  INFO("message: " << msg);
  CHECK(contains(msg, "structure 2 of 2"));
  CHECK(contains(msg, "'Ta3 coincident'"));
  CHECK(contains(msg, "atoms 1 and 3 (Ta, Ta)"));
  CHECK(contains(msg, "same position"));
}
