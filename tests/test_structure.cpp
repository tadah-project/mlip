#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
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
TEST_CASE( "Testing Structure virial pressure calculations", "[structure_virial_pressures]" ) {

  Structure st;
  st.cell.load(9.374769,0.0,0.0,
      0.0,9.374769,0.0,
      0.0,0.0,9.374769);

  REQUIRE(st.get_volume() == Approx(823.91));

  st.stress.load(257.0893807653,0.0,0.0,
      0.0,257.0893807653,0.0,
      0.0,0.0,257.0893807653);
  REQUIRE(fac*st.get_virial_pressure() == Approx(499.93));

}
TEST_CASE("Testing Structure read and write", "[structure_read_write]") {
    Structure st;
    st.read("tests_data/structure_1.dat");

    REQUIRE(st.get_volume() == Approx(989.521812));

    REQUIRE(fac * st.get_virial_pressure() == Approx(26.705578));
    REQUIRE(fac * st.get_pressure(300) == Approx(28.965914));
    REQUIRE(fac * st.get_pressure(0) == Approx(26.705578));

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

    REQUIRE(st_temp.get_volume() == Approx(989.521812));
    REQUIRE(fac * st_temp.get_virial_pressure() == Approx(26.705578));
    REQUIRE(fac * st_temp.get_pressure(300) == Approx(28.965914));
    REQUIRE(fac * st_temp.get_pressure(0) == Approx(26.705578));

    REQUIRE(st == st_temp);

    // Temporary file will be closed automatically
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
