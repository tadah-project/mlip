#include <string>
#include "catch2/catch.hpp"
#include "tadah/core/element.h"
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>

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
TEST_CASE( "Testing Structure read and write", "[structure_read_write]" ) {
  Structure st;
  st.read("tests_data/structure_1.dat");

  REQUIRE(st.get_volume() == Approx(989.521812));

  REQUIRE(fac*st.get_virial_pressure() == Approx(26.705578));
  REQUIRE(fac*st.get_pressure(300) == Approx(28.965914));
  REQUIRE(fac*st.get_pressure(0) == Approx(26.705578));

  std::string tempfile = std::tmpnam(nullptr);
  st.save(tempfile);

  Structure st_temp;
  st_temp.read(tempfile);

  REQUIRE(st_temp.get_volume() == Approx(989.521812));

  REQUIRE(fac*st_temp.get_virial_pressure() == Approx(26.705578));
  REQUIRE(fac*st_temp.get_pressure(300) == Approx(28.965914));
  REQUIRE(fac*st_temp.get_pressure(0) == Approx(26.705578));

  REQUIRE(st==st_temp);

  std::remove(tempfile.c_str());

}
TEST_CASE( "Testing Structure compare", "[structure_compare]" ) {
  Structure st;
  st.read("tests_data/structure_1.dat");
  std::string tempfile = std::tmpnam(nullptr);
  st.save(tempfile);

  Structure st_temp;
  st_temp.read(tempfile);

  SECTION("Compare unchanged") {
    REQUIRE(st==st_temp);
  }
  SECTION("Compare symbols") {
    REQUIRE(st==st_temp);
    st_temp.atoms[0].symbol[0]='X';
    st_temp.atoms[0].symbol[1]='X';
    REQUIRE(!(st==st_temp));
  }
  SECTION("Compare position") {
    REQUIRE(st==st_temp);
    st_temp.atoms[0].position(0.12,0.13,10.14);
    REQUIRE(!(st==st_temp));
  }
  SECTION("Compare force") {
    REQUIRE(st==st_temp);
    st_temp.atoms[0].force(1.12, 0.13, 0.134);
    REQUIRE(!(st==st_temp));
  }

  std::remove(tempfile.c_str());
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
