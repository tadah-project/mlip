// test_lammps_writer_cell.cpp
//
// `tadah data write -f lammps` (LammpsStructureWriter) writes a LAMMPS
// general-triclinic data file: the edge vectors A, B, C ("avec", "bvec",
// "cvec") are the rows a0, a1, a2 of Structure::cell, and the atoms keep
// their Cartesian positions. LAMMPS refuses edge vectors with
// (A x B) . C <= 0 ("must be right-handed", "co-planar"), so:
//   - a right-handed cell is written row by row, in order;
//   - a left-handed cell (valid in Tadah, e.g. read from CASTEP) is written
//     as avec = a0, bvec = a1, cvec = -a2: the same lattice, right-handed;
//   - a singular cell (zero or co-planar rows) is refused with
//     std::runtime_error naming the structure, before any file is written.
// The only other LAMMPS export test (test_dataset_precision.cpp) uses a
// diagonal cell, on which a transposed cell cannot be seen.

#include "catch2/catch.hpp"

#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/lammps_structure_writer.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Element;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using tadah::mlip::Atom;
using tadah::mlip::LammpsStructureWriter;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
namespace fs = std::filesystem;

struct LammpsCellTempDir {
  fs::path dir;
  LammpsCellTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_lmpcell_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~LammpsCellTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

// What a general-triclinic LAMMPS data file says: the three edge vectors
// by keyword, whether the "abc origin" line is present, and the atoms.
struct LammpsBox {
  std::map<std::string, Vec3d> edge;  // "avec", "bvec", "cvec"
  std::map<std::string, int> edge_count;
  int origin_lines = 0;
  std::vector<Vec3d> atoms;
};

LammpsBox parse_lammps_data(const std::string &file) {
  LammpsBox box;
  std::ifstream in(file);
  REQUIRE(in.good());
  std::string line;
  bool in_atoms = false;
  while (std::getline(in, line)) {
    if (line.rfind("Atoms", 0) == 0) { in_atoms = true; continue; }
    std::istringstream ls(line);
    if (in_atoms) {
      double id, type, x, y, z;
      if (ls >> id >> type >> x >> y >> z) box.atoms.emplace_back(x, y, z);
      continue;
    }
    double x, y, z;
    std::string tag, tag2;
    if (!(ls >> x >> y >> z >> tag)) continue;
    if (tag == "avec" || tag == "bvec" || tag == "cvec") {
      box.edge[tag] = Vec3d(x, y, z);
      box.edge_count[tag]++;
    } else if (tag == "abc" && (ls >> tag2) && tag2 == "origin") {
      box.origin_lines++;
    }
  }
  return box;
}

// Writes structure 0 of a one-structure database at 18 digits and parses it.
LammpsBox write_and_parse(const Structure &st, const LammpsCellTempDir &tmp,
                          const std::string &name) {
  StructureDB db;
  db.add(st);
  LammpsStructureWriter w(db);
  w.set_precision(18);
  const std::string file = tmp.path(name);
  w.write_data(file, 0);
  return parse_lammps_data(file);
}

double triple(const Vec3d &a, const Vec3d &b, const Vec3d &c) {
  return a * b.cross(c);
}

void check_vec(const Vec3d &got, const Vec3d &want) {
  for (int k = 0; k < 3; ++k) {
    INFO("component " << k);
    CHECK(got[k] == Approx(want[k]).margin(1e-14));
  }
}

// Two Ta atoms at fractional (0.1, 0.2, 0.3) and (0.6, 0.7, 0.4) of `cell`.
Structure two_atoms(const std::string &label, const Matrix3d &cell) {
  Structure st;
  st.label = label;
  st.cell = cell;
  const double f[2][3] = {{0.1, 0.2, 0.3}, {0.6, 0.7, 0.4}};
  for (const auto &fi : f) {
    Vec3d r;
    for (int k = 0; k < 3; ++k)
      r[k] = fi[0] * cell(0, k) + fi[1] * cell(1, k) + fi[2] * cell(2, k);
    st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0.0, 0.0, 0.0));
  }
  return st;
}

}  // namespace

TEST_CASE("LAMMPS data export writes a sheared cell's rows as avec, bvec, "
          "cvec in order",
          "[lammps_writer][cell_convention]") {
  // Right-handed cells: edge vector k is row k of the cell, exactly, and
  // the Cartesian positions are written unchanged.
  PeriodicTable::initialize();
  LammpsCellTempDir tmp;
  Matrix3d sheared;  // a0 = (2.5,0,0), a1 = (0.5,2,0), a2 = (0,0,3); det 15
  sheared.load(2.5, 0.0, 0.0,
               0.5, 2.0, 0.0,
               0.0, 0.0, 3.0);
  Matrix3d triclinic;  // every entry non-zero, det 70.61
  triclinic.load(4.2, 0.3, -0.2,
                 1.3, 3.9, 0.4,
                 -0.9, 1.1, 4.6);
  for (const auto &cell : {sheared, triclinic}) {
    const Structure st = two_atoms("right-handed", cell);
    const LammpsBox box = write_and_parse(st, tmp, "right.lmp");
    for (const char *tag : {"avec", "bvec", "cvec"}) {
      INFO(tag);
      REQUIRE(box.edge_count.count(tag) == 1);
      CHECK(box.edge_count.at(tag) == 1);
    }
    CHECK(box.origin_lines == 1);
    check_vec(box.edge.at("avec"), Vec3d(cell(0, 0), cell(0, 1), cell(0, 2)));
    check_vec(box.edge.at("bvec"), Vec3d(cell(1, 0), cell(1, 1), cell(1, 2)));
    check_vec(box.edge.at("cvec"), Vec3d(cell(2, 0), cell(2, 1), cell(2, 2)));
    REQUIRE(box.atoms.size() == 2);
    for (std::size_t n = 0; n < 2; ++n) check_vec(box.atoms[n], st(n).position);
  }
  // Hand values of the sheared rows: a transposed cell would give
  // avec = (2.5, 0.5, 0) and bvec = (0, 2, 0).
  const LammpsBox box = write_and_parse(two_atoms("sheared", sheared), tmp,
                                        "sheared.lmp");
  check_vec(box.edge.at("avec"), Vec3d(2.5, 0.0, 0.0));
  check_vec(box.edge.at("bvec"), Vec3d(0.5, 2.0, 0.0));
  check_vec(box.edge.at("cvec"), Vec3d(0.0, 0.0, 3.0));
}

TEST_CASE("LAMMPS data export writes a left-handed cell as avec, bvec, "
          "-cvec",
          "[lammps_writer][cell_handedness]") {
  // Left-handed rows are a valid Tadah cell but LAMMPS rejects them. The
  // export writes a0, a1, -a2: the same lattice, now right-handed, with the
  // same volume, and the atoms at the same Cartesian positions (LAMMPS
  // wraps periodic coordinates into its box).
  PeriodicTable::initialize();
  LammpsCellTempDir tmp;
  const double s = 4.9343;
  Matrix3d md_stress;  // the md_stress.castep rows, det = -2 s^3
  md_stress.load(s, s, 0.0,
                 s, 0.0, s,
                 0.0, s, s);
  Matrix3d swapped;  // the triclinic cell above with rows 1, 2 swapped
  swapped.load(4.2, 0.3, -0.2,
               -0.9, 1.1, 4.6,
               1.3, 3.9, 0.4);
  for (const auto &cell : {md_stress, swapped}) {
    const Vec3d a0(cell(0, 0), cell(0, 1), cell(0, 2));
    const Vec3d a1(cell(1, 0), cell(1, 1), cell(1, 2));
    const Vec3d a2(cell(2, 0), cell(2, 1), cell(2, 2));
    REQUIRE(triple(a0, a1, a2) < 0.0);
    const Structure st = two_atoms("left-handed", cell);
    const LammpsBox box = write_and_parse(st, tmp, "left.lmp");
    REQUIRE(box.edge.size() == 3);
    check_vec(box.edge.at("avec"), a0);
    check_vec(box.edge.at("bvec"), a1);
    check_vec(box.edge.at("cvec"), -a2);
    const double t = triple(box.edge.at("avec"), box.edge.at("bvec"),
                            box.edge.at("cvec"));
    CHECK(t > 0.0);
    CHECK(t == Approx(-triple(a0, a1, a2)));
    REQUIRE(box.atoms.size() == 2);
    for (std::size_t n = 0; n < 2; ++n) check_vec(box.atoms[n], st(n).position);
  }
}

TEST_CASE("LAMMPS data export refuses a singular cell",
          "[lammps_writer][cell_handedness]") {
  // A zero cell (a non-periodic structure) or co-planar rows cannot be a
  // LAMMPS box. The writer throws std::runtime_error naming the structure
  // and writes no file (it used to write a data file LAMMPS rejects).
  PeriodicTable::initialize();
  LammpsCellTempDir tmp;
  Matrix3d coplanar;  // a2 = a0 + a1
  coplanar.load(3.0, 0.0, 0.0,
                0.0, 3.0, 0.0,
                3.0, 3.0, 0.0);
  struct Case { const char *label; Matrix3d cell; };
  for (const Case &c : {Case{"zero-cell", Matrix3d()},
                        Case{"coplanar-cell", coplanar}}) {
    INFO(c.label);
    Structure st;
    st.label = c.label;
    st.cell = c.cell;
    st.add_atom(Atom(Element("Ta"), 0.5, 0.5, 0.5, 0.0, 0.0, 0.0));
    StructureDB db;
    db.add(st);
    LammpsStructureWriter w(db);
    const std::string file = tmp.path(std::string(c.label) + ".lmp");
    CHECK_THROWS_AS(w.write_data(file, 0), std::runtime_error);
    CHECK_THROWS_WITH(w.write_data(file, 0), Catch::Contains(c.label));
    CHECK_FALSE(fs::exists(file));
  }
}
