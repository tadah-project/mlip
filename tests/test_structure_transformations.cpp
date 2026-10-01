// test_structure_transformation.cpp
#include "catch2/catch.hpp"
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/structure_properties.h>
#include <tadah/mlip/atom.h>
#include <tadah/core/periodic_table.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>

using namespace tadah::core;
using namespace tadah::mlip;
// Helper function to build a simple 2×2×2 box with two atoms.
static Structure makeSimpleStructure()
{
    // 2x2x2 cell => volume 8 Å^3
    Structure st;
    st.cell.set_zero();
    st.cell(0,0) = 2.0;
    st.cell(1,1) = 2.0;
    st.cell(2,2) = 2.0;
    st.stress.set_zero();
    st.label = "Simple";

    // Add H at (0,0,0) and He at (1,1,1)
    Atom h(PeriodicTable::find_by_symbol("H"), 0.0, 0.0, 0.0);
    Atom he(PeriodicTable::find_by_symbol("He"), 1.0, 1.0, 1.0);
    st.add_atom(h);
    st.add_atom(he);

    return st;
}

// Pins scale(): a geometric scaling of the cell and the positions that
// leaves every stored label as it was. The scaled structure is a different
// structure, and its stored energy, forces and stress no longer belong to it
// (the class documentation says so); scale() does not pretend to transform
// them, so the stress, like the energy and the forces, is unchanged
// (scaleToVolume() goes through scale()). The stress is the Cauchy stress in
// eV/A^3; multiplying it by factor^2 has no physical basis.
// Before the fix: scale(st, 2) multiplied the stress by factor^2, giving
// 4.0 and 6.0 instead of 1.0 and 1.5 (and 4x the off-diagonal -0.25).
TEST_CASE("StructureTransformation scale()", "[transform]")
{
    Structure st = makeSimpleStructure();

    SECTION("Invalid scale factor throws std::invalid_argument") {
        REQUIRE_THROWS_AS(StructureTransformation::scale(st, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(StructureTransformation::scale(st, -1.0), std::invalid_argument);
    }

    SECTION("Valid scale factor scales positions and cell; labels unchanged") {
        st.stress(0,0) = 1.0;
        st.stress(1,1) = 1.5;
        st.stress(1,2) = st.stress(2,1) = -0.25;
        st.energy = -7.5;
        st.atoms[1].force = Vec3d(0.3, -0.2, 0.1);

        double oldVol = StructureProperties::getVolume(st);
        REQUIRE(oldVol == Approx(8.0));

        StructureTransformation::scale(st, 2.0);

        double newVol = StructureProperties::getVolume(st);
        REQUIRE(newVol == Approx(64.0));

        // Positions doubled
        REQUIRE(st.atoms[0].position[0] == Approx(0.0));
        REQUIRE(st.atoms[1].position[0] == Approx(2.0));

        // Stress, energy and forces are left as they were (exactly).
        CHECK(st.stress(0,0) == 1.0);
        CHECK(st.stress(1,1) == 1.5);
        CHECK(st.stress(2,2) == 0.0);
        CHECK(st.stress(1,2) == -0.25);
        CHECK(st.stress(2,1) == -0.25);
        CHECK(st.stress(0,1) == 0.0);
        CHECK(st.energy == -7.5);
        CHECK(st.atoms[1].force[0] == 0.3);
        CHECK(st.atoms[1].force[1] == -0.2);
        CHECK(st.atoms[1].force[2] == 0.1);
    }

    SECTION("scaleToVolume() leaves the stress unchanged") {
        st.stress(0,0) = 1.0;
        st.stress(1,1) = 1.5;
        StructureTransformation::scaleToVolume(st, 27.0);
        REQUIRE(StructureProperties::getVolume(st) == Approx(27.0).epsilon(1e-12));
        CHECK(st.stress(0,0) == 1.0);
        CHECK(st.stress(1,1) == 1.5);
    }
}

// Pins rescaleLength(s), the LSCALE transform: the same configuration seen at
// lengths times s, E'(r') = E(r'/s). Positions and cell are multiplied by s,
// the energy is unchanged, forces are divided by s (chain rule), and the
// stored stress -- the Cauchy stress (1/V) dE/d(strain) in eV/A^3 -- is
// divided by s^3: the strain derivative of E is unchanged and V' = s^3 V.
// Equivalently the energy-unit virial V*sigma is invariant.
// Reference values by hand: 1.05^3 = 1.157625, so 1.0, 1.5, -0.4, -0.25
// become 0.863837598531476, 1.29575639779721, -0.345535039412590,
// -0.215959399632869; the cube a = 3 A with sigma = 0.01 I at s = 2 gives
// 0.01 / 8 = 0.00125 (V = 27 -> 216 A^3, V*sigma = 0.27 eV both times).
// Before the fix: rescaleLength() left the stress unchanged (the stored
// stress was taken for an energy-unit virial), so the cases below read 1.0,
// 1.5, -0.4, -0.25 and 0.01 instead.
TEST_CASE("StructureTransformation rescaleLength()", "[transform]")
{
    Structure st = makeSimpleStructure();

    SECTION("Invalid rescale factor throws std::invalid_argument") {
        REQUIRE_THROWS_AS(StructureTransformation::rescaleLength(st, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(StructureTransformation::rescaleLength(st, -1.0), std::invalid_argument);
    }

    SECTION("s = 1 is identity") {
        // give atoms forces and stress so we can detect changes
        st.atoms[0].force = Vec3d(1.0, 2.0, 3.0);
        st.atoms[1].force = Vec3d(-0.5, 0.0, 0.5);
        st.stress(0,0) = 1.0;
        st.stress(1,2) = -0.25;
        st.energy = -7.0;

        const Vec3d pos0_before = st.atoms[0].position;
        const Vec3d pos1_before = st.atoms[1].position;
        const Matrix3d cell_before = st.cell;

        StructureTransformation::rescaleLength(st, 1.0);

        REQUIRE(st.cell(0,0) == Approx(cell_before(0,0)));
        REQUIRE(st.cell(1,1) == Approx(cell_before(1,1)));
        REQUIRE(st.cell(2,2) == Approx(cell_before(2,2)));
        REQUIRE(st.atoms[0].position[0] == Approx(pos0_before[0]));
        REQUIRE(st.atoms[1].position[0] == Approx(pos1_before[0]));
        REQUIRE(st.atoms[0].force[0] == Approx(1.0));
        REQUIRE(st.atoms[0].force[1] == Approx(2.0));
        REQUIRE(st.atoms[0].force[2] == Approx(3.0));
        REQUIRE(st.atoms[1].force[2] == Approx(0.5));
        REQUIRE(st.stress(0,0) == Approx(1.0));
        REQUIRE(st.stress(1,2) == Approx(-0.25));
        REQUIRE(st.energy == Approx(-7.0));
    }

    SECTION("Length rescale transforms positions/cell/forces/stress; energy invariant") {
        st.atoms[0].force = Vec3d(1.0, 2.0, 3.0);
        st.atoms[1].force = Vec3d(-0.5, 0.0, 0.5);
        st.stress.set_zero();
        st.stress(0,0) = 1.0;
        st.stress(1,1) = 1.5;
        st.stress(2,2) = -0.4;
        st.stress(0,2) = st.stress(2,0) = -0.25;
        st.energy = -7.5;

        const double s = 1.05;
        StructureTransformation::rescaleLength(st, s);

        // cell scaled
        REQUIRE(st.cell(0,0) == Approx(2.0 * s));
        REQUIRE(st.cell(1,1) == Approx(2.0 * s));
        REQUIRE(st.cell(2,2) == Approx(2.0 * s));

        // positions scaled
        REQUIRE(st.atoms[0].position[0] == Approx(0.0));
        REQUIRE(st.atoms[1].position[0] == Approx(1.0 * s));
        REQUIRE(st.atoms[1].position[1] == Approx(1.0 * s));
        REQUIRE(st.atoms[1].position[2] == Approx(1.0 * s));

        // forces divided by s
        REQUIRE(st.atoms[0].force[0] == Approx(1.0 / s));
        REQUIRE(st.atoms[0].force[1] == Approx(2.0 / s));
        REQUIRE(st.atoms[0].force[2] == Approx(3.0 / s));
        REQUIRE(st.atoms[1].force[0] == Approx(-0.5 / s));
        REQUIRE(st.atoms[1].force[2] == Approx(0.5 / s));

        // Cauchy stress (eV/A^3) divided by s^3 = 1.157625, every component
        CHECK(st.stress(0,0) == Approx(0.863837598531476).epsilon(1e-13));
        CHECK(st.stress(1,1) == Approx(1.29575639779721).epsilon(1e-13));
        CHECK(st.stress(2,2) == Approx(-0.345535039412590).epsilon(1e-13));
        CHECK(st.stress(0,2) == Approx(-0.215959399632869).epsilon(1e-13));
        CHECK(st.stress(2,0) == Approx(-0.215959399632869).epsilon(1e-13));
        CHECK(st.stress(0,1) == 0.0);
        CHECK(st.stress(1,2) == 0.0);

        // energy invariant
        REQUIRE(st.energy == Approx(-7.5));

        // volume scales by s^3, so the virial V*sigma is invariant
        const double newVol = StructureProperties::getVolume(st);
        REQUIRE(newVol == Approx(8.0 * s * s * s));
        CHECK(newVol * st.stress(0,0) == Approx(8.0 * 1.0).epsilon(1e-13));
        CHECK(newVol * st.stress(2,2) == Approx(8.0 * -0.4).epsilon(1e-13));
    }

    SECTION("Cube a = 3 A, sigma = 0.01 I, s = 2: sigma' = 0.00125") {
        Structure cube;
        cube.cell.set_zero();
        cube.cell(0,0) = cube.cell(1,1) = cube.cell(2,2) = 3.0;
        cube.stress.set_zero();
        cube.stress(0,0) = cube.stress(1,1) = cube.stress(2,2) = 0.01;
        cube.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 0.0, 0.0, 0.0));
        cube.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 1.5, 1.5, 1.5));
        cube.atoms[1].force = Vec3d(0.2, 0.0, -0.4);

        StructureTransformation::rescaleLength(cube, 2.0);

        REQUIRE(StructureProperties::getVolume(cube) == Approx(216.0).epsilon(1e-14));
        for (int i = 0; i < 3; ++i) {
            INFO("diagonal " << i);
            CHECK(cube.stress(i,i) == Approx(0.00125).epsilon(1e-14));
        }
        CHECK(cube.stress(0,1) == 0.0);
        CHECK(cube.atoms[1].force[0] == Approx(0.1).epsilon(1e-14));
        CHECK(cube.atoms[1].force[2] == Approx(-0.2).epsilon(1e-14));
        CHECK(cube.atoms[1].position[0] == Approx(3.0).epsilon(1e-14));
    }

    SECTION("Round trip s and 1/s recovers original") {
        st.atoms[0].force = Vec3d(1.0, 2.0, 3.0);
        st.stress(0,0) = 1.234;
        st.energy = -3.14;

        const Vec3d pos1_before = st.atoms[1].position;
        const Matrix3d cell_before = st.cell;
        const double s = 1.05;

        StructureTransformation::rescaleLength(st, s);
        StructureTransformation::rescaleLength(st, 1.0 / s);

        REQUIRE(st.cell(0,0) == Approx(cell_before(0,0)));
        REQUIRE(st.atoms[1].position[0] == Approx(pos1_before[0]));
        REQUIRE(st.atoms[0].force[0] == Approx(1.0));
        REQUIRE(st.stress(0,0) == Approx(1.234));
        REQUIRE(st.energy == Approx(-3.14));
    }
}

TEST_CASE("StructureTransformation scaleToVolume()", "[transform]")
{
    Structure st = makeSimpleStructure();
    double initVol = StructureProperties::getVolume(st);
    REQUIRE(initVol == Approx(8.0));

    SECTION("Throws if target volume <= 0") {
        REQUIRE_THROWS_AS(StructureTransformation::scaleToVolume(st, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(StructureTransformation::scaleToVolume(st, -10.0), std::invalid_argument);
    }

    SECTION("Scales to new volume") {
        double targetVol = 125.0; // 5×5×5
        StructureTransformation::scaleToVolume(st, targetVol);
        double newVol = StructureProperties::getVolume(st);
        REQUIRE(newVol == Approx(targetVol).margin(1e-10));
    }
}

TEST_CASE("StructureTransformation scaleVolumeToDensity()", "[transform]")
{
    // This structure has 2 atoms: H (~1.008 amu), He (~4.0026 amu).
    // total mass ~5.0106 amu => ~8.31567e-24 g
    // volume=8.0 Å^3 => ~8.0e-24 cm^3 => init density ~1.04 g/cm^3
    Structure st = makeSimpleStructure();

    SECTION("Throws if target density <= 0") {
        REQUIRE_THROWS_AS(StructureTransformation::scaleVolumeToDensity(st, 0.0), std::invalid_argument);
        REQUIRE_THROWS_AS(StructureTransformation::scaleVolumeToDensity(st, -1.0), std::invalid_argument);
    }

    SECTION("Scales to a target density") {
        double targetDensity = 2.0; // g/cm^3
        StructureTransformation::scaleVolumeToDensity(st, targetDensity);
        double newDensity = StructureProperties::getDensity(st);
        REQUIRE(newDensity == Approx(targetDensity).margin(1e-3));
    }
}

TEST_CASE("StructureTransformation rotate()", "[transform]")
{
    Structure st = makeSimpleStructure();
    st.stress(0,0) = 1.0;
    st.stress(1,1) = 2.0;
    st.stress(2,2) = 3.0;

    // 90° rotation about z-axis
    Matrix3d R;
    R.set_zero();
    R(0,1) = -1.0;
    R(1,0) = 1.0;
    R(2,2) = 1.0;

    StructureTransformation::rotate(st, R);

    // positions: H(0,0,0)->(0,0,0), He(1,1,1)->(-1,1,1)
    REQUIRE(st.atoms[1].position[0] == Approx(-1.0));
    REQUIRE(st.atoms[1].position[1] == Approx(1.0));
    REQUIRE(st.atoms[1].position[2] == Approx(1.0));

    // Volume unchanged
    double vol = StructureProperties::getVolume(st);
    REQUIRE(vol == Approx(8.0));
}

TEST_CASE("StructureTransformation shear()", "[transform]")
{
    Structure st = makeSimpleStructure();
    // Shear matrix => x' = x + 0.5y
    Matrix3d shearMat;
    shearMat.set_zero();
    shearMat(0,0) = 1.0;
    shearMat(1,1) = 1.0;
    shearMat(2,2) = 1.0;
    shearMat(0,1) = 0.5;

    double oldVol = StructureProperties::getVolume(st);
    StructureTransformation::shear(st, shearMat);

    double newVol = StructureProperties::getVolume(st);
    REQUIRE(oldVol == Approx(8.0));
    REQUIRE(newVol == Approx(8.0));
}

TEST_CASE("StructureTransformation replicate()", "[transform]")
{
    Structure st = makeSimpleStructure();
    REQUIRE(st.natoms() == 2);

    // replicate(2,1,1)
    StructureTransformation::replicate(st, 2,1,1);
    REQUIRE(st.natoms() == 4);

    double vol = StructureProperties::getVolume(st);
    REQUIRE(vol == Approx(16.0)); // 4×2×2 => 16

    REQUIRE(st.cell(0,0) == Approx(4.0));
    REQUIRE(st.cell(1,1) == Approx(2.0));
    REQUIRE(st.cell(2,2) == Approx(2.0));
}

TEST_CASE("StructureTransformation shift()", "[transform]")
{
    Structure st = makeSimpleStructure();
    Vec3d shiftVec(1.0, -2.0, 0.5);

    StructureTransformation::shift(st, shiftVec);
    REQUIRE(st.atoms[0].position[0] == Approx(1.0));
    REQUIRE(st.atoms[1].position[1] == Approx(-1.0));
}

TEST_CASE("StructureTransformation mirrorOrigin()", "[transform]")
{
    Structure st = makeSimpleStructure();
    StructureTransformation::mirrorOrigin(st);
    // H(0,0,0)->(0,0,0), He(1,1,1)->(-1,-1,-1)
    REQUIRE(st.atoms[1].position[0] == Approx(-1.0));
    REQUIRE(st.atoms[1].position[1] == Approx(-1.0));
    REQUIRE(st.atoms[1].position[2] == Approx(-1.0));
}

TEST_CASE("StructureTransformation mirrorPlane()", "[transform]")
{
    // Mirror about plane normal to x => x-> -x
    Structure st = makeSimpleStructure();
    Vec3d n(1.0, 0.0, 0.0);
    StructureTransformation::mirrorPlane(st, n);
    REQUIRE(st.atoms[1].position[0] == Approx(-1.0));
    REQUIRE(st.atoms[1].position[1] == Approx(1.0));
}

TEST_CASE("StructureTransformation mirrorHkl()", "[transform]")
{
    // Mirror about (1,0,0) => effectively x-> -x for a simple diagonal cell
    Structure st = makeSimpleStructure();
    SECTION("Invalid (0,0,0) => throw") {
        REQUIRE_THROWS_AS(StructureTransformation::mirrorHkl(st, 0,0,0), std::invalid_argument);
    }
    SECTION("Mirror along (1,0,0)") {
        StructureTransformation::mirrorHkl(st, 1,0,0);
        REQUIRE(st.atoms[1].position[0] == Approx(-1.0));
    }
}

TEST_CASE("StructureTransformation remapToCell()", "[transform]")
{
    Structure st = makeSimpleStructure();
    // place He outside => (2.5, -0.5, 3.0)
    st.atoms[1].position = Vec3d(2.5, -0.5, 3.0);

    StructureTransformation::remapToCell(st);
    // expected => (0.5,1.5,1.0)
    REQUIRE(st.atoms[1].position[0] == Approx(0.5));
    REQUIRE(st.atoms[1].position[1] == Approx(1.5));
    REQUIRE(st.atoms[1].position[2] == Approx(1.0));
}

TEST_CASE("StructureTransformation addAtom()", "[transform]")
{
    Structure st = makeSimpleStructure();
    REQUIRE(st.natoms() == 2);

    SECTION("Add atom at fixed position") {
        Atom c(PeriodicTable::find_by_symbol("C"), 0,0,0);
        StructureTransformation::addAtom(st, c, false, Vec3d(3.0,4.0,5.0));
        REQUIRE(st.natoms() == 3);
        REQUIRE(st.atoms.back().position[0] == Approx(3.0));
        REQUIRE(st.atoms.back().Z == 6); // Carbon
    }

    SECTION("Add atom randomly") {
        Atom c(PeriodicTable::find_by_symbol("C"), 0,0,0);
        StructureTransformation::addAtom(st, c, true);
        auto &pos = st.atoms.back().position;
        // Can't be exact, but must lie within [0,2]
        REQUIRE(pos[0] >= 0.0);   REQUIRE(pos[0] < 2.0);
        REQUIRE(pos[1] >= 0.0);   REQUIRE(pos[1] < 2.0);
        REQUIRE(pos[2] >= 0.0);   REQUIRE(pos[2] < 2.0);
    }
}

TEST_CASE("StructureTransformation removeAtom()", "[transform]")
{
    Structure st = makeSimpleStructure();
    REQUIRE(st.natoms() == 2);

    SECTION("Valid index") {
        StructureTransformation::removeAtom(st, 1);
        REQUIRE(st.natoms() == 1);
    }

    SECTION("Invalid index => no throw, prints warning") {
        REQUIRE_NOTHROW(StructureTransformation::removeAtom(st, 999));
        REQUIRE(st.natoms() == 2);
    }
}

TEST_CASE("StructureTransformation shakeAtoms()", "[transform]")
{
    Structure st = makeSimpleStructure();

    SECTION("Non-positive maxDisplacement => no action") {
        StructureTransformation::shakeAtoms(st, 0.0);
        REQUIRE(st.atoms[0].position[0] == Approx(0.0));
        REQUIRE(st.atoms[1].position[0] == Approx(1.0));
    }

    SECTION("Positive displacement => positions change randomly") {
        StructureTransformation::shakeAtoms(st, 0.5);
        // Each coordinate of each atom should be within ±0.5 of original
        for (size_t i=0; i<st.natoms(); i++) {
            for (int d=0; d<3; d++) {
                double val = st.atoms[i].position[d];
                double expectedMin = double(i) - 0.5;
                double expectedMax = double(i) + 0.5;
                REQUIRE(val >= Approx(expectedMin).margin(0.001));
                REQUIRE(val <= Approx(expectedMax).margin(0.001));
            }
        }
    }
}

// Helper to create a 3-atom structure in a non-cubic cell for more robust checking.
static Structure makeTrioStructure()
{
    // Create a 3×3×2 cell => volume = 3*3*2 = 18 Å^3
    // Place 3 atoms in somewhat triangular arrangement.

    Structure st;
    st.cell.set_zero();
    st.cell(0,0) = 3.0;   // x-axis
    st.cell(1,1) = 3.0;   // y-axis
    st.cell(2,2) = 2.0;   // z-axis
    st.stress.set_zero();
    st.label = "TrioBox";

    // Suppose we have H at (0.0, 0.0, 0.0),
    // He at (1.5, 1.5, 1.0),
    // C at (2.9, 2.5, 0.5)
    // At least one of them is near the boundary to test partial wrapping or expansions.
    Atom h(PeriodicTable::find_by_symbol("H"), 0.0, 0.0, 0.0);
    Atom he(PeriodicTable::find_by_symbol("He"), 1.5, 1.5, 1.0);
    Atom c (PeriodicTable::find_by_symbol("C"),  2.9, 2.5, 0.5);

    st.add_atom(h);
    st.add_atom(he);
    st.add_atom(c);

    return st;
}

TEST_CASE("StructureTransformation replicate() more complex", "[transform][extended]")
{
    // We replicate a 3-atom system in a 2×2×1 manner => 3×2×2×1 = 12 atoms total
    Structure st = makeTrioStructure();
    REQUIRE(st.natoms() == 3);

    double oldVolume = StructureProperties::getVolume(st);
    REQUIRE(oldVolume == Approx(18.0).margin(1e-12));

    // replicate(2,2,1)
    StructureTransformation::replicate(st, 2, 2, 1);
    REQUIRE(st.natoms() == 3 * 2 * 2 * 1);

    double newVolume = StructureProperties::getVolume(st);
    // The original cell is 3×3×2 => replicate in x=2 => new x=6, in y=2 => new y=6, z unchanged => new z=2
    // => 6×6×2 = 72
    REQUIRE(newVolume == Approx(72.0).margin(1e-12));
}

TEST_CASE("StructureTransformation remapToCell() wide out-of-box atom", "[transform][extended]")
{
    // Place an atom far outside the box in negative/positive directions
    Structure st = makeTrioStructure();
    // Insert a new atom at e.g. (-5, 10, 4). This is well outside the 3×3×2 cell.
    Atom extra(PeriodicTable::find_by_symbol("O"), -5.0, 10.0, 4.0);
    st.add_atom(extra);
    REQUIRE(st.natoms() == 4);

    // Now remap
    StructureTransformation::remapToCell(st);

    // The cell is 3 in x, 3 in y, 2 in z
    // We'll convert fractional => xFrac = (-5 / 3) => -1.666..., wrap => 0.333... => final => ~1.0
    // yFrac = 10/3 => 3.333..., wrap => 0.333..., => final => ~1.0
    // zFrac = 4/2 => 2.0 => wrap => 0.0 => final => 0.0
    // => final ~ (1.0, 1.0, 0.0)
    Vec3d pos = st.atoms.back().position;
    REQUIRE(pos[0] == Approx(1.0).margin(1e-6));
    REQUIRE(pos[1] == Approx(1.0).margin(1e-6));
    REQUIRE(pos[2] == Approx(0.0).margin(1e-6));
}

TEST_CASE("StructureTransformation mirrorHkl() with (1,1,0) plane", "[transform][extended]")
{
    // We'll test reflection about a plane that is normal to (1,1,0) in reciprocal-lattice space.
    // For our trio cell => a1=(3,0,0), a2=(0,3,0), a3=(0,0,2)
    // (h,k,l)=(1,1,0) => normal is basically along the sum of b1+b2 => x+y direction in direct space.
    Structure st = makeTrioStructure();
    REQUIRE_NOTHROW(StructureTransformation::mirrorHkl(st, 1,1,0));

    // We can't easily do a direct numeric check unless we replicate the reflection math,
    // but we can at least check volume remains the same magnitude (reflection => det < 0, but |det| is same).
    double vol = StructureProperties::getVolume(st);
    REQUIRE(std::fabs(vol) == Approx(18.0).margin(1e-12));

    // Also confirm that the third atom (near x=2.9,y=2.5) now has x or y negated in some consistent reflection sense.
    // We'll do a sanity check that the new positions differ from old by more than some threshold
    const auto &a2 = st.atoms[2].position; 
    // It's presumably "mirrored" in a plane with normal ~ (1/sqrt(2),1/sqrt(2),0).
    // We won't match exact numbers, just confirm it's not the same as original 2.9,2.5,0.5
    REQUIRE(std::fabs(a2[0] - 2.9) > 0.1);
    REQUIRE(std::fabs(a2[1] - 2.5) > 0.1);
}

TEST_CASE("StructureTransformation shakeAtoms() large triple-atom structure", "[transform][extended]")
{
    // Similar to existing test, but we do it for a 3-atom structure with bigger displacement
    Structure st = makeTrioStructure();
    double maxDisp = 1.0;  // more significant shake

    StructureTransformation::shakeAtoms(st, maxDisp);

    // Each original atom was at (0,0,0), (1.5,1.5,1.0), (2.9,2.5,0.5).
    // Now each coordinate is in [orig - 1.0, orig + 1.0].
    for (size_t i=0; i < st.natoms(); ++i) {
        const Vec3d &pos = st.atoms[i].position;
        // We'll do a quick bounding check
        // e.g., for the second atom's x=1.5 => must be in [0.5, 2.5]
        // We'll just do a generic approach using the original positions from makeTrioStructure()
        // For each dimension:
        //   low = orig - 1.0
        //   high = orig + 1.0
        // We'll store them in an array for quick reference:
    }

    // Let's re-implement the bounding logic more explicitly:
    std::vector<Vec3d> originalPositions{
        {0.0, 0.0, 0.0},
        {1.5, 1.5, 1.0},
        {2.9, 2.5, 0.5}
    };
    for (size_t i = 0; i < st.natoms(); i++) {
        double ox = originalPositions[i][0];
        double oy = originalPositions[i][1];
        double oz = originalPositions[i][2];

        Vec3d &curr = st.atoms[i].position;
        // Check if inside [ox - 1.0, ox + 1.0], etc.
        REQUIRE(curr[0] >= Approx(ox - 1.0).margin(1e-7));
        REQUIRE(curr[0] <= Approx(ox + 1.0).margin(1e-7));
        REQUIRE(curr[1] >= Approx(oy - 1.0).margin(1e-7));
        REQUIRE(curr[1] <= Approx(oy + 1.0).margin(1e-7));
        REQUIRE(curr[2] >= Approx(oz - 1.0).margin(1e-7));
        REQUIRE(curr[2] <= Approx(oz + 1.0).margin(1e-7));
    }
}

TEST_CASE("StructureTransformation addAtom() checks minDistance in a 3-atom structure", "[transform][extended]")
{
    Structure st = makeTrioStructure();
    REQUIRE(st.natoms() == 3);

    // We'll attempt to add a new H atom near the second atom => we expect a warning
    // The code won't throw, but we can verify it doesn't fail catastrophically.
    // We'll do some extremely close position => e.g. (1.45,1.45,1.05)
    Atom newH(PeriodicTable::find_by_symbol("H"), 0.0,0.0,0.0); // template
    Vec3d closePos(1.45, 1.45, 1.05);
    REQUIRE_NOTHROW(StructureTransformation::addAtom(st, newH, false, closePos, 0.5));
    // We expect a console warning but no exception. Now there's 4 atoms
    REQUIRE(st.natoms() == 4);

    // The newly added atom's position is set to closePos
    const auto &lastPos = st.atoms.back().position;
    REQUIRE(lastPos[0] == Approx(1.45).margin(1e-6));
    REQUIRE(lastPos[1] == Approx(1.45).margin(1e-6));
    REQUIRE(lastPos[2] == Approx(1.05).margin(1e-6));
}

TEST_CASE("StructureTransformation::center() now calls either getCentreOfMass or getGeometricCenter", "[transform][new]") 
{
    // We'll do a 2-atom system again, to illustrate difference.
    Structure st;
    st.cell.set_zero();
    st.cell(0,0) = 1.0;
    st.cell(1,1) = 1.0;
    st.cell(2,2) = 1.0;
    Atom h(PeriodicTable::find_by_symbol("H"), 0.0, 0.0, 0.0);
    Atom he(PeriodicTable::find_by_symbol("He"), 1.0,  1.0, 1.0);
    st.add_atom(h);
    st.add_atom(he);

    SECTION("Center by geometric center => He => (0.5,0.5,0.5)") {
        StructureTransformation::center(st, /*useCenterOfMass=*/false, /*shiftCell=*/false);
        // Atom 0 => old(0,0,0) - (0.5,0.5,0.5) => (-0.5,-0.5,-0.5)
        // Atom 1 => old(1,1,1) - (0.5,0.5,0.5) => (0.5,0.5,0.5)
        REQUIRE(st.atoms[0].position[0] == Approx(-0.5));
        REQUIRE(st.atoms[1].position[0] == Approx(0.5));
    }

    SECTION("Center by center of mass => He => ~ (0.2,0.2,0.2)") {
        // We'll create a fresh structure since the previous one is offset
        Structure st2;
        st2.cell.set_zero();
        st2.cell(0,0) = 1.0;
        st2.cell(1,1) = 1.0;
        st2.cell(2,2) = 1.0;
        st2.add_atom(h);
        st2.add_atom(he);

        StructureTransformation::center(st2, /*useCenterOfMass=*/true, /*shiftCell=*/false);

        // The heavier He is ~ 4x the mass => new He pos = old(1,1,1) - COM => Should be around (0.2,0.2,0.2)
        // Let's check the actual final position:
        Vec3d hePos = st2.atoms[1].position;
        REQUIRE(hePos[0] == Approx(0.2).margin(0.1));
        REQUIRE(hePos[1] == Approx(0.2).margin(0.1));
        REQUIRE(hePos[2] == Approx(0.2).margin(0.1));
    }
}
