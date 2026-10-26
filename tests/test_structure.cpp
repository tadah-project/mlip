#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include "catch2/catch.hpp"
#include "tadah/core/element.h"
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_table.h>
#include <tadah/mlip/structure.h>

using namespace tadah::core;
using namespace tadah::mlip;
// Conversion factor from eV/A^3 to kbar;
double fac = 1602.1766208;


TEST_CASE("Testing Structure PeriodicTable Initialization") {
    PeriodicTable::initialize();
}
TEST_CASE( "Testing Structure class volume", "[structure_volume]" ) {
  //using vec=Eigen::Vector3d;

  //std::string symbol="Ti";
  //std::string name="Titanium";
  //int Z = 22;
  //vec pos(1.0,2.0,3.0);
  //vec force(4.0,5.0,6.0);
  //Element element(symbol,name,Z);

  //Atom a(element, 1.0, 2.0, 3.0, 
  //        4.0, 5.0, 6.0);

  Structure st;
  // Integer volume
  st.cell.load(2,-1,3,
      3,2,-4,
      -2,0,1);

  REQUIRE(st.get_volume() == 11 );

  // double volume
  st.cell.load(5.0147,5.0104,-5.0018,
      -9.9924,-3.3395,-9.9662,
      -25.6396,38.4594,12.8198);

  REQUIRE(st.get_volume() == Approx(5980.0279772134).epsilon(1e-10));


}
// Pins the pressure helpers under the stored stress convention: Structure::
// stress is the Cauchy stress in eV/A^3, positive under tension, so
// get_virial_pressure() = -tr(sigma)/3 in eV/A^3 (positive under
// compression; the stored stress is already per volume, so the volume does
// not enter) and get_pressure(T) adds the ideal-gas term N kB T / V.
// Reference values by hand (exact decimal arithmetic):
//  - sigma = -0.312030 I (compression) gives P = +0.312030 eV/A^3, i.e.
//    0.312030 * 1602.1766208 = +499.927170988224 kbar (the 499.93 kbar this
//    case has always quoted), whatever the volume;
//  - a = 3 A cube (V = 27 A^3), sigma = +0.01 I (tension) gives
//    P = -0.01 eV/A^3 (-1.60 GPa); with 2 atoms at T = 300 K,
//    N kB T / V = 2 * 8.617333262145e-5 * 300 / 27 = 0.00191496294714333,
//    so get_pressure(300) = -0.00808503705285667 eV/A^3.
// The tolerances (relative 1e-12, 1e-14 where no volume enters) cover the
// rounding of a trace, a cube and one product.
// Before the fix: get_virial_pressure() returned +tr(sigma)/(3V), the
// pressure of an energy-unit "virial" V*sigma: -0.312030/823.9137 =
// -3.787e-4 (eV/A^6, wrong sign) = -0.6068 "kbar", and +3.704e-4 instead of
// -0.01 for the tensile cube.
TEST_CASE( "Testing Structure virial pressure calculations", "[structure_virial_pressures]" ) {

  Structure st;
  st.cell.load(9.374769,0.0,0.0,
      0.0,9.374769,0.0,
      0.0,0.0,9.374769);

  REQUIRE(st.get_volume() == Approx(823.913702672641).epsilon(1e-12));

  st.stress.load(-0.312030,0.0,0.0,
      0.0,-0.312030,0.0,
      0.0,0.0,-0.312030);
  CHECK(st.get_virial_pressure() == Approx(0.312030).epsilon(1e-14));
  CHECK(fac*st.get_virial_pressure() == Approx(499.927170988224).epsilon(1e-12));
  CHECK(fac*st.get_virial_pressure() == Approx(499.93).margin(0.005));

  // Only the trace enters: a symmetric shear leaves the pressure unchanged.
  st.stress(0,1) = st.stress(1,0) = 0.05;
  st.stress(1,2) = st.stress(2,1) = -0.02;
  st.stress(0,2) = st.stress(2,0) = 0.01;
  CHECK(st.get_virial_pressure() == Approx(0.312030).epsilon(1e-14));

  // Tension: a positive Cauchy stress is a negative pressure.
  Structure cube;
  cube.cell.load(3.0,0.0,0.0,
      0.0,3.0,0.0,
      0.0,0.0,3.0);
  cube.stress.load(0.01,0.0,0.0,
      0.0,0.01,0.0,
      0.0,0.0,0.01);
  cube.add_atom(Atom(Element("Ta"),0.0,0.0,0.0,0,0,0));
  cube.add_atom(Atom(Element("Ta"),1.5,1.5,1.5,0,0,0));
  REQUIRE(cube.get_volume() == Approx(27.0).epsilon(1e-14));
  CHECK(cube.get_virial_pressure() == Approx(-0.01).epsilon(1e-14));
  CHECK(cube.get_pressure(0) == Approx(-0.01).epsilon(1e-14));
  CHECK(cube.get_pressure(300) ==
        Approx(-0.00808503705285667).epsilon(1e-12));
}
// Pins tests_data/structure_1.dat and the pressure of a real structure.
// The fixture is Structure 13 of Tadah.MLIP/examples/CLI/ex_1/tdata.db
// (LAMMPS MD with the Ta1_Ravelo_2013 EAM, 54 atoms), copied exactly: its
// cell, energy, Cauchy stress in eV/A^3 and all three force components; only
// the label and the weights line (1.1 0.12 0.765e-8, which exercises the
// three-weight form) are the test's own. The structure is under tension:
// tr(sigma)/3 = +0.0166683108833 eV/A^3, so
//   get_virial_pressure() = -0.0166683108833333 eV/A^3 = -26.7055780055029 kbar,
//   get_pressure(300)     = that + 54 kB 300 / 989.521820687578
//                         = -0.0152575203789891 eV/A^3 = -24.4452424425958 kbar
// (exact decimal arithmetic on the file's values; relative tolerance 1e-12).
// Before the fix: get_virial_pressure() returned +tr(sigma)/(3V) =
// +1.6845e-5 (+0.0270 "kbar"), and get_pressure(300) +2.2873 "kbar". (The
// old fixture stored V*sigma, which made the old formula print +26.7056, the
// pressure with the wrong sign, and its z-force column repeated fy.)
TEST_CASE("Testing Structure read and write", "[structure_read_write]") {
    Structure st;
    st.read("tests_data/structure_1.dat");

    // The fixture is the source structure, not V*sigma, and fz is not fy.
    REQUIRE(st.natoms() == 54);
    CHECK(st.energy == -433.794159295855);
    CHECK(st.stress(0, 0) == 0.017265228156);
    CHECK(st.stress(1, 1) == 0.015028531649);
    CHECK(st.stress(2, 2) == 0.017711172845);
    CHECK(st.stress(0, 1) == 0.000675090722);
    CHECK(st.stress(1, 2) == -0.000079905262);
    CHECK(st.stress(2, 0) == 0.000748708098);
    CHECK(st.atoms[0].force(1) == -0.132338283317);
    CHECK(st.atoms[0].force(2) == -0.691587489453);
    CHECK(st.atoms[53].force(2) == -0.626870790718);

    REQUIRE(st.get_volume() == Approx(989.521820687578).epsilon(1e-12));

    CHECK(st.get_virial_pressure() ==
          Approx(-0.0166683108833333).epsilon(1e-12));
    CHECK(fac * st.get_virial_pressure() ==
          Approx(-26.7055780055029).epsilon(1e-12));
    CHECK(fac * st.get_pressure(300) ==
          Approx(-24.4452424425958).epsilon(1e-12));
    CHECK(fac * st.get_pressure(0) ==
          Approx(-26.7055780055029).epsilon(1e-12));

    // Create a temporary file
    std::FILE* tmpFile = std::tmpfile();
    if (!tmpFile) {
        FAIL("Unable to create temporary file");
    }

    const int fd = fileno(tmpFile);
    REQUIRE(fd != -1);

    char tempfile[256];
    snprintf(tempfile, sizeof(tempfile), "/proc/self/fd/%d", fd);

    st.save(tempfile);

    Structure st_temp;
    st_temp.read(tempfile);

    std::fclose(tmpFile);

    CHECK(st_temp.get_volume() == Approx(989.521820687578).epsilon(1e-12));
    CHECK(fac * st_temp.get_virial_pressure() ==
          Approx(-26.7055780055029).epsilon(1e-12));
    CHECK(fac * st_temp.get_pressure(300) ==
          Approx(-24.4452424425958).epsilon(1e-12));
    CHECK(fac * st_temp.get_pressure(0) ==
          Approx(-26.7055780055029).epsilon(1e-12));
    CHECK(st_temp.atoms[0].force(2) == -0.691587489453);

    REQUIRE(st == st_temp);
}
TEST_CASE("Testing Structure compare", "[structure_compare]") {
    Structure st;
    st.read("tests_data/structure_1.dat");

    // Create a temporary file
    std::FILE* tmpFile = std::tmpfile();
    if (!tmpFile) {
        FAIL("Unable to create temporary file");
    }

    const int fd = fileno(tmpFile);
    REQUIRE(fd != -1);

    char tempfile[256];
    snprintf(tempfile, sizeof(tempfile), "/proc/self/fd/%d", fd);

    st.save(tempfile);

    Structure st_temp;
    st_temp.read(tempfile);

    SECTION("Compare unchanged") {
        REQUIRE(st == st_temp);
    }
    SECTION("Compare symbols") {
        REQUIRE(st == st_temp);
        st_temp.atoms[0].symbol[0] = 'X';
        st_temp.atoms[0].symbol[1] = 'X';
        REQUIRE(!(st == st_temp));
    }
    SECTION("Compare position") {
        REQUIRE(st == st_temp);
        st_temp.atoms[0].position(0.12, 0.13, 10.14);
        REQUIRE(!(st == st_temp));
    }
    SECTION("Compare force") {
        REQUIRE(st == st_temp);
        st_temp.atoms[0].force(1.12, 0.13, 0.134);
        REQUIRE(!(st == st_temp));
    }

    // Temporary file will be closed automatically
}
TEST_CASE( "Testing Structure copy", "[structure_copy]" ) {
  Structure st;
  st.read("tests_data/structure_1.dat");
  Structure st2=st;
  REQUIRE(st==st2);
}
TEST_CASE("Testing Structure Unique Element") {
  Structure st;
  st.add_atom(Atom(Element("Ta"),0,0,0,0,0,0));
  REQUIRE(st.get_unique_elements().size()==1);
}
TEST_CASE("Structure::save -> read round-trips every double bit-exactly",
          "[structure_read_write][structure_save]") {
  // Full-precision values, small magnitudes and fz != fy: the old writer
  // kept 6 significant digits for weights/energy, 6 fixed decimals for the
  // rest, and wrote force(1) twice instead of force(2).
  Structure st;
  st.label = "save round trip";
  st.eweight = 8.100000072900001e-05;
  st.fweight = 1.0 / 3.0;
  st.sweight = 7.65e-9;
  st.energy = -12345.678901234567;
  st.cell.load(9.9321641234567, 1e-7, -0.1,
               0.30000000000000004, 14.87106012345678, 2.0 / 3.0,
               -1e-20, 0.0, 17.525592123456789);
  st.stress.load(3.3333333333333335e-05, 1e-20, -0.1,
                 1e-20, 257.0893807653, 1.0 / 7.0,
                 -0.1, 1.0 / 7.0, -123456.789012345);
  st.add_atom(Atom(Element("Ta"), 9.9321641234567, -0.004522, 1e-7,
                   1e-7, 0.5, -0.3));
  st.add_atom(Atom(Element("H"), 1.0 / 3.0, 2.0 / 3.0, 0.1,
                   -1.2345678901234566e-07, 0.25, 0.75));

  std::FILE* tmpFile = std::tmpfile();
  REQUIRE(tmpFile);
  const int fd = fileno(tmpFile);
  REQUIRE(fd != -1);
  char tempfile[256];
  snprintf(tempfile, sizeof(tempfile), "/proc/self/fd/%d", fd);

  st.save(tempfile);
  Structure rt;
  rt.read(tempfile);
  std::fclose(tmpFile);

  CHECK(rt.label == st.label);
  CHECK(rt.eweight == st.eweight);
  CHECK(rt.fweight == st.fweight);
  CHECK(rt.sweight == st.sweight);
  CHECK(rt.energy == st.energy);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("i=" << i << " j=" << j);
      CHECK(rt.cell(i, j) == st.cell(i, j));
      CHECK(rt.stress(i, j) == st.stress(i, j));
    }
  REQUIRE(rt.natoms() == st.natoms());
  for (size_t a = 0; a < st.natoms(); ++a) {
    INFO("atom " << a);
    CHECK(rt.atoms[a].symbol == st.atoms[a].symbol);
    for (int k = 0; k < 3; ++k) {
      INFO("k=" << k);
      CHECK(rt.atoms[a].position(k) == st.atoms[a].position(k));
      CHECK(rt.atoms[a].force(k) == st.atoms[a].force(k));
    }
  }
  CHECK(rt.atoms[0].force(2) != rt.atoms[0].force(1));
  CHECK(rt == st);
}

TEST_CASE("Structure::save keeps the caller's stream format and reports "
          "an unwritable path", "[structure_save]") {
  Structure st;
  st.label = "fmt";
  st.add_atom(Atom(Element("Ta"), 0.5, 0.25, 0.125, 0, 0, 0));

  std::FILE* tmpFile = std::tmpfile();
  REQUIRE(tmpFile);
  char tempfile[256];
  snprintf(tempfile, sizeof(tempfile), "/proc/self/fd/%d", fileno(tmpFile));
  std::ofstream ofs(tempfile);
  REQUIRE(ofs.is_open());
  ofs << std::fixed << std::setprecision(3);
  st.save(ofs);
  CHECK((ofs.flags() & std::ios_base::floatfield) == std::ios_base::fixed);
  CHECK(ofs.precision() == 3);
  ofs.close();
  std::fclose(tmpFile);

  CHECK_THROWS_AS(st.save("/nonexistent_dir_for_tadah_test/x.dat"),
                  std::runtime_error);
}

/* Pins: printing a Structure (operator<<, which tadah data print writes to
 * standard output) leaves the format of the stream as it found it: after
 * a structure is written to a stream of the default format, its flags,
 * precision, width and fill are those it had, and 1.0 then prints "1"
 * (a kernel's settings line copies the format of std::cout, and a program
 * may run several commands in one process).
 * Before the fix: the stream was left std::fixed and std::left at the
 * printer's precision, and 1.0 printed "1.000000". */
TEST_CASE("Printing a Structure leaves the format of the stream as it "
          "found it", "[structure][stream_state]") {
  PeriodicTable::initialize();
  Structure st;
  st.label = "fmt";
  st.cell.load(3.3, 0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 3.3);
  st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0, 0, 0));
  st.add_atom(Atom(Element("Ta"), 1.65, 1.65, 1.65, 0, 0, 0));
  std::ostringstream os;
  const auto flags = os.flags();
  const auto precision = os.precision();
  const auto fill = os.fill();
  os << st;
  CHECK(os.str().find("Structure  :    fmt") != std::string::npos);
  CHECK(os.flags() == flags);
  CHECK(os.precision() == precision);
  CHECK(os.width() == 0);
  CHECK(os.fill() == fill);
  std::ostringstream after;
  after.copyfmt(os);
  after << 1.0;
  CHECK(after.str() == "1");
}

// get_nn_iindex(i, j, jj) returns the position ii of the reverse entry
// (i, -s) in atom j's list for the entry jj = (j, s) of atom i. A table
// without that reverse entry is broken (every NNFinder table is symmetric),
// and returning idx[j].size() made callers read one past the end of j's
// list. The lookup reports the broken table instead.
TEST_CASE("get_nn_iindex throws when the reverse entry is missing",
          "[structure][nn]") {
  PeriodicTable::initialize();
  Structure st;
  st.cell.load(4.0, 0.0, 0.0,
               0.0, 4.0, 0.0,
               0.0, 0.0, 4.0);
  st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0, 0, 0));
  st.add_atom(Atom(Element("Ta"), 2.0, 2.0, 2.0, 0, 0, 0));

  // disp[i][k] = s0 a0 + s1 a1 + s2 a2 for the cube above.
  auto entry = [](NNTable &t, size_t i, size_t j, double s0, double s1,
                  double s2) {
    t.idx[i].push_back(j);
    t.shift[i].push_back(Vec3d(s0, s1, s2));
    t.disp[i].push_back(Vec3d(4.0 * s0, 4.0 * s1, 4.0 * s2));
  };
  auto empty_table = []() {
    auto t = std::make_shared<NNTable>();
    t->idx.resize(2);
    t->shift.resize(2);
    t->disp.resize(2);
    return t;
  };

  SECTION("a symmetric table: the reverse entry is found") {
    auto t = empty_table();
    entry(*t, 0, 1, 0.0, 0.0, 0.0);
    entry(*t, 0, 1, -1.0, 0.0, 0.0);
    entry(*t, 1, 0, 0.0, 0.0, 0.0);
    entry(*t, 1, 0, 1.0, 0.0, 0.0);
    st.nn = t;
    CHECK(st.get_nn_iindex(0, 1, 0) == 0);
    CHECK(st.get_nn_iindex(0, 1, 1) == 1);
    CHECK(st.get_nn_iindex(1, 0, 0) == 0);
    CHECK(st.get_nn_iindex(1, 0, 1) == 1);
  }

  SECTION("atom 1 has no entry for atom 0") {
    auto t = empty_table();
    entry(*t, 0, 1, 0.0, 0.0, 0.0);
    st.nn = t;
    CHECK_THROWS_AS(st.get_nn_iindex(0, 1, 0), std::logic_error);
  }

  SECTION("atom 1 lists atom 0 only at a different image") {
    // The reverse of (0 -> 1, s = (1,0,0)) is (1 -> 0, s = (-1,0,0));
    // atom 1 holds (0, (1,0,0)) instead, which is the reverse of a
    // different entry.
    auto t = empty_table();
    entry(*t, 0, 1, 1.0, 0.0, 0.0);
    entry(*t, 1, 0, 1.0, 0.0, 0.0);
    st.nn = t;
    CHECK_THROWS_AS(st.get_nn_iindex(0, 1, 0), std::logic_error);
    CHECK_THROWS_AS(st.get_nn_iindex(1, 0, 0), std::logic_error);
  }
}
