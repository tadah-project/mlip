// test_cell_builder.cpp
//
// Phase D2 (PC_OBSERVABLE redesign): coverage for the native cell
// builder primitives in tadah::mlip::cells (MLIP/include/tadah/mlip/
// cell_builder.h). These are the building blocks the cohesive-energy
// observable composes with NativeRelaxer to replace LAMMPS scripts.

#include "catch2/catch.hpp"

#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_transformations.h>

#include <cmath>
#include <stdexcept>

namespace {
constexpr double kTol = 1e-12;
}

TEST_CASE("cells::bcc builds conventional 2-atom unit cell",
          "[mlip][cell_builder][bcc]")
{
    tadah::core::PeriodicTable::initialize();

    const double a0 = 3.30;
    auto st = tadah::mlip::cells::bcc("Ta", a0);

    REQUIRE(st.natoms() == 2);

    /* Cell vectors must be a0 * I — orthorhombic cubic. */
    CHECK(std::abs(st.cell(0, 0) - a0)   < kTol);
    CHECK(std::abs(st.cell(0, 1))         < kTol);
    CHECK(std::abs(st.cell(0, 2))         < kTol);
    CHECK(std::abs(st.cell(1, 0))         < kTol);
    CHECK(std::abs(st.cell(1, 1) - a0)   < kTol);
    CHECK(std::abs(st.cell(1, 2))         < kTol);
    CHECK(std::abs(st.cell(2, 0))         < kTol);
    CHECK(std::abs(st.cell(2, 1))         < kTol);
    CHECK(std::abs(st.cell(2, 2) - a0)   < kTol);

    /* Atom positions: corner + body-centre. */
    CHECK(std::abs(st(0).position[0])         < kTol);
    CHECK(std::abs(st(0).position[1])         < kTol);
    CHECK(std::abs(st(0).position[2])         < kTol);
    CHECK(std::abs(st(1).position[0] - 0.5 * a0) < kTol);
    CHECK(std::abs(st(1).position[1] - 0.5 * a0) < kTol);
    CHECK(std::abs(st(1).position[2] - 0.5 * a0) < kTol);

    /* Both atoms are the requested species. */
    CHECK(st(0).Z == tadah::core::Element("Ta").Z);
    CHECK(st(1).Z == tadah::core::Element("Ta").Z);

    /* Volume sanity. */
    CHECK(std::abs(st.get_volume() - a0 * a0 * a0) < 1e-10);
}

TEST_CASE("cells::bcc rejects non-positive a0",
          "[mlip][cell_builder][bcc][error]")
{
    CHECK_THROWS_AS(tadah::mlip::cells::bcc("Ta", 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(tadah::mlip::cells::bcc("Ta", -1.0),
                    std::invalid_argument);
}

TEST_CASE("cells::bcc composes with Structure::replicate for a supercell",
          "[mlip][cell_builder][bcc][supercell]")
{
    tadah::core::PeriodicTable::initialize();

    const double a0 = 3.30;
    auto st = tadah::mlip::cells::bcc("Ta", a0);

    /* Replicate 4×4×4: matches BASINSv2_blip6/Ecoh_fast.in's
     * `region box block 0 4 0 4 0 4 units lattice` (which generates 128
     * atoms with `lattice bcc a0`). */
    tadah::mlip::StructureTransformation::replicate(st, 4, 4, 4);
    CHECK(st.natoms() == 4 * 4 * 4 * 2);

    /* Replicated cell edges. */
    CHECK(std::abs(st.cell(0, 0) - 4.0 * a0) < 1e-9);
    CHECK(std::abs(st.cell(1, 1) - 4.0 * a0) < 1e-9);
    CHECK(std::abs(st.cell(2, 2) - 4.0 * a0) < 1e-9);
}

TEST_CASE("cells::isolated_atom builds a single atom in a vacuum box",
          "[mlip][cell_builder][isolated_atom]")
{
    tadah::core::PeriodicTable::initialize();

    const double box = 20.0;
    auto st = tadah::mlip::cells::isolated_atom("Ta", box);

    REQUIRE(st.natoms() == 1);

    CHECK(std::abs(st.cell(0, 0) - box) < kTol);
    CHECK(std::abs(st.cell(1, 1) - box) < kTol);
    CHECK(std::abs(st.cell(2, 2) - box) < kTol);
    CHECK(st(0).Z == tadah::core::Element("Ta").Z);
    /* Atom placed at the centre of the box. */
    CHECK(std::abs(st(0).position[0] - 0.5 * box) < kTol);
    CHECK(std::abs(st(0).position[1] - 0.5 * box) < kTol);
    CHECK(std::abs(st(0).position[2] - 0.5 * box) < kTol);
}

TEST_CASE("cells::isolated_atom default box_size = 20",
          "[mlip][cell_builder][isolated_atom][default]")
{
    tadah::core::PeriodicTable::initialize();
    auto st = tadah::mlip::cells::isolated_atom("H");
    CHECK(std::abs(st.cell(0, 0) - 20.0) < kTol);
}

TEST_CASE("cells::isolated_atom rejects non-positive box_size",
          "[mlip][cell_builder][isolated_atom][error]")
{
    CHECK_THROWS_AS(tadah::mlip::cells::isolated_atom("H", 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(tadah::mlip::cells::isolated_atom("H", -3.0),
                    std::invalid_argument);
}

TEST_CASE("cells::fcc and cells::hcp throw not-implemented in MVP",
          "[mlip][cell_builder][stub]")
{
    CHECK_THROWS_AS(tadah::mlip::cells::fcc("Ta", 3.7),
                    std::logic_error);
    CHECK_THROWS_AS(tadah::mlip::cells::hcp("Ta", 3.0),
                    std::logic_error);
}
