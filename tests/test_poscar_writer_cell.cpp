// test_poscar_writer_cell.cpp
//
// `tadah data write -f VASP` (VaspPoscarWriter) writes the rows a0, a1, a2
// of Structure::cell as the three lattice lines and the atoms as Cartesian
// coordinates. VASP stops on a left-handed basis ("the triple product of the
// basis vectors is negative"), so a left-handed cell (valid in Tadah, e.g.
// read from CASTEP) is written as a0, a1, -a2: the same lattice, right-handed,
// with the atoms at the same Cartesian positions. A singular cell is refused
// with std::runtime_error naming the structure, before any file is written.
// The file is read back with VaspPoscarReader and parsed by hand.

#include "catch2/catch.hpp"

#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>
#include <tadah/mlip/structure_writers/vasp_poscar_writer.h>

#include <cmath>
#include <filesystem>
#include <fstream>
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
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::VaspPoscarReader;
using tadah::mlip::VaspPoscarWriter;
namespace fs = std::filesystem;

struct PoscarWriterTempDir {
  fs::path dir;
  PoscarWriterTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_poscarcell_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~PoscarWriterTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

double triple(const Vec3d &a, const Vec3d &b, const Vec3d &c) {
  return a * b.cross(c);
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

// The lattice lines (3-5) of a written POSCAR, and its scale line.
struct PoscarText {
  double scale = 0.0;
  Vec3d rows[3];
};

PoscarText parse_lattice(const std::string &file) {
  PoscarText p;
  std::ifstream in(file);
  REQUIRE(in.good());
  std::string line;
  std::getline(in, line);  // comment
  std::getline(in, line);
  p.scale = std::stod(line);
  for (int i = 0; i < 3; ++i) {
    std::getline(in, line);
    std::istringstream ls(line);
    double x, y, z;
    REQUIRE(static_cast<bool>(ls >> x >> y >> z));
    p.rows[i] = Vec3d(x, y, z);
  }
  return p;
}

std::string write_poscar(const Structure &st, const PoscarWriterTempDir &tmp,
                         const std::string &name) {
  StructureDB db;
  db.add(st);
  VaspPoscarWriter w(db);
  w.set_precision(18);
  const std::string file = tmp.path(name);
  w.write_data(file, 0);
  return file;
}

void check_vec(const Vec3d &got, const Vec3d &want) {
  for (int k = 0; k < 3; ++k) {
    INFO("component " << k);
    CHECK(got[k] == Approx(want[k]).margin(1e-14));
  }
}

}  // namespace

TEST_CASE("POSCAR export writes a right-handed cell's rows as its lattice "
          "lines and reads back unchanged",
          "[poscar_writer][cell_convention]") {
  PeriodicTable::initialize();
  PoscarWriterTempDir tmp;
  Matrix3d triclinic;  // every entry non-zero, det 70.61
  triclinic.load(4.2, 0.3, -0.2,
                 1.3, 3.9, 0.4,
                 -0.9, 1.1, 4.6);
  const Structure st = two_atoms("right-handed", triclinic);
  const std::string file = write_poscar(st, tmp, "POSCAR_right");
  const PoscarText p = parse_lattice(file);
  CHECK(p.scale == 1.0);
  for (int i = 0; i < 3; ++i)
    check_vec(p.rows[i], Vec3d(triclinic(i, 0), triclinic(i, 1), triclinic(i, 2)));
  VaspPoscarReader r;
  r.read(file);
  const Structure back = r.getStructure();
  REQUIRE(back.natoms() == 2);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) CHECK(back.cell(i, k) == Approx(triclinic(i, k)));
  for (std::size_t n = 0; n < 2; ++n) check_vec(back(n).position, st(n).position);
}

TEST_CASE("POSCAR export writes a left-handed cell as a0, a1, -a2",
          "[poscar_writer][cell_handedness]") {
  // VASP refuses a negative triple product of the lattice lines; the export
  // writes the same lattice right-handed, and the Cartesian positions are
  // unchanged (VASP maps them into the cell).
  PeriodicTable::initialize();
  PoscarWriterTempDir tmp;
  const double s = 4.9343;
  Matrix3d md_stress;  // the md_stress.castep rows, det = -2 s^3
  md_stress.load(s, s, 0.0,
                 s, 0.0, s,
                 0.0, s, s);
  const Vec3d a0(s, s, 0.0), a1(s, 0.0, s), a2(0.0, s, s);
  REQUIRE(triple(a0, a1, a2) < 0.0);
  const Structure st = two_atoms("left-handed", md_stress);
  const std::string file = write_poscar(st, tmp, "POSCAR_left");
  const PoscarText p = parse_lattice(file);
  check_vec(p.rows[0], a0);
  check_vec(p.rows[1], a1);
  check_vec(p.rows[2], -a2);
  CHECK(triple(p.rows[0], p.rows[1], p.rows[2]) > 0.0);
  VaspPoscarReader r;
  r.read(file);
  const Structure back = r.getStructure();
  REQUIRE(back.natoms() == 2);
  CHECK(back.get_volume() == Approx(2.0 * s * s * s));
  for (std::size_t n = 0; n < 2; ++n) check_vec(back(n).position, st(n).position);
}

TEST_CASE("POSCAR export refuses a singular cell",
          "[poscar_writer][cell_handedness]") {
  PeriodicTable::initialize();
  PoscarWriterTempDir tmp;
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
    VaspPoscarWriter w(db);
    const std::string file = tmp.path(std::string(c.label) + ".vasp");
    CHECK_THROWS_AS(w.write_data(file, 0), std::runtime_error);
    CHECK_THROWS_WITH(w.write_data(file, 0), Catch::Contains(c.label));
    CHECK_FALSE(fs::exists(file));
  }
}
