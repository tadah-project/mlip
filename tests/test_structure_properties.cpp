#include "catch2/catch.hpp"
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_properties.h>
#include <tadah/mlip/atom.h>
#include <tadah/core/periodic_table.h>

using namespace tadah::core;
using namespace tadah::mlip;
// Helper to create an element from symbol
static Element makeElement(const std::string& sym) {
    return PeriodicTable::find_by_symbol(sym);
}

TEST_CASE("StructureProperties: getVolume() with default/zeroed cell", "[structure_props][new]") {
    Structure st;
    // By default, st.cell is zero-initialized -> volume = 0
    REQUIRE(StructureProperties::getVolume(st) == Approx(0.0));
}

TEST_CASE("StructureProperties: getVolume() with known cell", "[structure_props][new]") {
    Structure st;

    // Let's define a simple 3x3 identity cell => volume = 1 Å^3
    st.cell(0, 0) = 1.0;  st.cell(0, 1) = 0.0;  st.cell(0, 2) = 0.0;
    st.cell(1, 0) = 0.0;  st.cell(1, 1) = 1.0;  st.cell(1, 2) = 0.0;
    st.cell(2, 0) = 0.0;  st.cell(2, 1) = 0.0;  st.cell(2, 2) = 1.0;

    double vol = StructureProperties::getVolume(st);
    REQUIRE(vol == Approx(1.0));
}

TEST_CASE("StructureProperties: getDensity() with zero-volume cell", "[structure_props][new]") {
    Structure st;
    // By default, st.cell is zero => volume is 0 => getDensity returns 0.0
    // We also add a few atoms to ensure it doesn't crash.
    Atom a1(makeElement("H"), 0.0, 0.0, 0.0,0,0,0);
    Atom a2(makeElement("C"), 1.0, 0.0, 0.0,0,0,0);
    st.add_atom(a1);
    st.add_atom(a2);

    double rho = StructureProperties::getDensity(st);
    REQUIRE(rho == Approx(0.0)); // The code is written to return 0 on <=0 volume
}

TEST_CASE("StructureProperties: getDensity() with known cell and atoms", "[structure_props][new]") {
    // We'll define a 10Å x 10Å x 10Å cubic cell => volume = 1000 Å^3
    // Add 2 atoms: H (1 amu) and He (4 amu). Total = ~5 amu
    // volume in cm^3 = 1000 * 1e-24 = 1e-21 cm^3
    // mass in grams = 5 amu * 1.66053906660e-24 g/amu = ~8.3026953e-24 g
    // => density = (8.3026953e-24 g) / (1e-21 cm^3) = 8.3026953e-03 g/cm^3
    // => 0.008320297
    Structure st;

    // 10x10x10 cubic
    st.cell(0,0) = 10.0;  st.cell(0,1) = 0.0;   st.cell(0,2) = 0.0;
    st.cell(1,0) = 0.0;   st.cell(1,1) = 10.0;  st.cell(1,2) = 0.0;
    st.cell(2,0) = 0.0;   st.cell(2,1) = 0.0;   st.cell(2,2) = 10.0;

    Atom h(makeElement("H"), 0.0, 0.0, 0.0,0,0,0);
    Atom he(makeElement("He"), 5.0, 5.0, 5.0,0,0,0);
    st.add_atom(h);
    st.add_atom(he);

    double rho = StructureProperties::getDensity(st);
    REQUIRE(rho == Approx(0.008320297).margin(1e-7));
}

TEST_CASE("StructureProperties: getCentreOfMass() with no atoms", "[structure_props][new]") {
    Structure st;
    // No atoms => centre = (0,0,0) by definition
    Vec3d com = StructureProperties::getCentreOfMass(st);
    REQUIRE(com[0] == Approx(0.0));
    REQUIRE(com[1] == Approx(0.0));
    REQUIRE(com[2] == Approx(0.0));
}

TEST_CASE("StructureProperties: getCentreOfMass() with single atom", "[structure_props][new]") {
    // If one atom at (1,2,3), CoM should be (1,2,3)
    Structure st;
    Atom a(makeElement("C"), 1.0, 2.0, 3.0,0,0,0); // Carbon, but mass won't matter for single-atom
    st.add_atom(a);

    Vec3d com = StructureProperties::getCentreOfMass(st);
    REQUIRE(com[0] == Approx(1.0));
    REQUIRE(com[1] == Approx(2.0));
    REQUIRE(com[2] == Approx(3.0));
}

TEST_CASE("StructureProperties: getCentreOfMass() with multiple atoms", "[structure_props][new]") {
    // Suppose we place 2 identical atoms of mass M at:
    //  ( 2, 0, 0 ) and ( -2, 0, 0 )
    // Then CoM = ( (M*(2) + M*(-2))/ (2M), 0, 0 ) = (0,0,0).
    // We'll pick O for both, but in practice same mass anyway.
    Structure st;
    st.cell.set_zero(); // not used for CoM
    Atom O1(makeElement("O"), 2.0, 0.0, 0.0,0,0,0);
    Atom O2(makeElement("O"), -2.0, 0.0, 0.0,0,0,0);
    st.add_atom(O1);
    st.add_atom(O2);

    Vec3d com = StructureProperties::getCentreOfMass(st);
    REQUIRE(com[0] == Approx(0.0));
    REQUIRE(com[1] == Approx(0.0));
    REQUIRE(com[2] == Approx(0.0));
}

TEST_CASE("StructureProperties: getCentreOfMass() with different masses", "[structure_props][new]") {
    // Let's put H (mass ~1) at (0,0,0)  and O (mass ~16) at (0, 3, 0)
    // total mass ~17
    // Weighted sum of positions = 1*(0,0,0) + 16*(0,3,0) = (0,48,0)
    // => CoM = (0,48,0)/17 ~ (0,2.823529...,0)
    Structure st;

    Atom H(makeElement("H"), 0.0, 0.0, 0.0,0,0,0);
    Atom O(makeElement("O"), 0.0, 3.0, 0.0,0,0,0);
    st.add_atom(H);
    st.add_atom(O);

    Vec3d com = StructureProperties::getCentreOfMass(st);
    // 48 / 17 ~ 2.823529...
    REQUIRE(com[0] == Approx(0.0));
    REQUIRE(com[1] == Approx(2.8221908626).margin(1e-8));
    REQUIRE(com[2] == Approx(0.0));
}

TEST_CASE("StructureProperties getGeometricCenter() vs. getCentreOfMass()", "[properties][new]") 
{
    // We'll build a small structure with 2 atoms:
    //   H(0,0,0) => mass ~1.008
    //   He(1,1,1) => mass ~4.0026
    // The geometric center => (0.5, 0.5, 0.5).
    // The mass-weighted center => ~ ( (4.0026*1,1,1) + (1.008*0,0,0 ) ) / 5.0106 => ~ (0.80,0.80,0.80).
    Structure st;
    st.cell.set_zero();
    st.cell(0,0) = 1.0;
    st.cell(1,1) = 1.0;
    st.cell(2,2) = 1.0;

    Atom h(PeriodicTable::find_by_symbol("H"), 0.0, 0.0, 0.0);
    Atom he(PeriodicTable::find_by_symbol("He"), 1.0, 1.0, 1.0);
    st.add_atom(h);
    st.add_atom(he);

    Vec3d geom = StructureProperties::getGeometricCenter(st);
    Vec3d com  = StructureProperties::getCentreOfMass(st);

    // Geometric center => (0.5,0.5,0.5)
    REQUIRE(geom[0] == Approx(0.5));
    REQUIRE(geom[1] == Approx(0.5));
    REQUIRE(geom[2] == Approx(0.5));

    // COM => near (0.80,0.80,0.80)
    REQUIRE(com[0] == Approx(0.80).margin(0.03));
    REQUIRE(com[1] == Approx(0.80).margin(0.03));
    REQUIRE(com[2] == Approx(0.80).margin(0.03));
}

