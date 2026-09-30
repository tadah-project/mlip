/**
 * Test: NNFinder cutoff equivalence
 *
 * Verifies that computing descriptors with a large NN cutoff (7.0 Å) and
 * filtering to a smaller RCUTMAX (4.5 Å) gives identical results to
 * computing with the smaller cutoff (4.5 Å) directly.
 *
 * This is the key correctness property needed for HPO: val_stdb_ NN is built
 * at the maximum possible cutoff, then each iteration's DescriptorsCalc
 * filters at the actual optimised cutoff. If the lists are not equivalent,
 * HPO's reported validation metrics will disagree with predict_engine.
 *
 * Two representative cells are tested:
 *  - 5.9 Å cubic  (the cell height is above 4.5 Å and below 7.0 Å, so the
 *                  7.0 Å list holds images beyond the nearest cells:
 *                  check_box is TRUE at 4.5 Å and FALSE at 7.0 Å)
 *  - 10.0 Å cubic (the cell height is above both cutoffs)
 */

#include "tadah_test_skip.hpp"  // before catch2/catch.hpp (it needs Catch's
                                 // listener interface)
#include "catch2/catch.hpp"
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/context.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace tadah::core;
using namespace tadah::mlip;

// ---------------------------------------------------------------------------
// Helper: build a CH4-like structure in a cubic cell of side 'a'.
// One C at the centre, four H at tetrahedral positions (bond length 1.09 Å).
// ---------------------------------------------------------------------------
static Structure build_ch4(double a)
{
    PeriodicTable::initialize();

    Structure st;
    st.cell(0,0) = a;  st.cell(0,1) = 0;  st.cell(0,2) = 0;
    st.cell(1,0) = 0;  st.cell(1,1) = a;  st.cell(1,2) = 0;
    st.cell(2,0) = 0;  st.cell(2,1) = 0;  st.cell(2,2) = a;

    const double cx = a*0.5, cy = a*0.5, cz = a*0.5;
    const double b  = 1.09 / std::sqrt(3.0);   // offset per axis for tetrahedral H

    // C at centre
    st.add_atom(Atom(PeriodicTable::find_by_symbol("C"), cx, cy, cz, 0,0,0));
    // 4 H at tetrahedral positions (wrap into [0,a) if needed)
    auto w = [&](double x){ return x < 0 ? x+a : (x >= a ? x-a : x); };
    st.add_atom(Atom(PeriodicTable::find_by_symbol("H"), w(cx+b), w(cy+b), w(cz+b), 0,0,0));
    st.add_atom(Atom(PeriodicTable::find_by_symbol("H"), w(cx+b), w(cy-b), w(cz-b), 0,0,0));
    st.add_atom(Atom(PeriodicTable::find_by_symbol("H"), w(cx-b), w(cy+b), w(cz-b), 0,0,0));
    st.add_atom(Atom(PeriodicTable::find_by_symbol("H"), w(cx-b), w(cy-b), w(cz+b), 0,0,0));

    return st;
}

// ---------------------------------------------------------------------------
// Helper: build a StructureDB with N copies of the same structure.
// ---------------------------------------------------------------------------
static StructureDB build_stdb(const Structure& proto, int n = 1)
{
    StructureDB stdb;
    for (int i = 0; i < n; ++i)
        stdb.structures.push_back(proto);
    return stdb;
}

// ---------------------------------------------------------------------------
// Helper: look up (neighbor_idx, shift) in the NNTable of atom i.
// Returns true if found.
// ---------------------------------------------------------------------------
static bool has_nn_entry(const Structure& st, size_t i,
                         size_t nbr_idx, const Vec3d& shift)
{
    const auto& nn = *st.nn;
    for (size_t k = 0; k < nn.idx[i].size(); ++k) {
        if (nn.idx[i][k] == nbr_idx) {
            const Vec3d& s = nn.shift[i][k];
            if (std::fabs(s[0]-shift[0]) < 0.5 &&
                std::fabs(s[1]-shift[1]) < 0.5 &&
                std::fabs(s[2]-shift[2]) < 0.5)
                return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Core check: every (nbr,shift) entry from the small-cutoff NN list must
// also appear in the large-cutoff NN list.
// Returns a non-empty error string if the check fails.
// ---------------------------------------------------------------------------
static std::string check_superset(const Structure& st_small,
                                  const Structure& st_large,
                                  double rcut_small)
{
    const double rcut_small_sq = rcut_small * rcut_small;
    const auto& small = *st_small.nn;
    std::ostringstream err;

    for (size_t i = 0; i < st_small.natoms(); ++i) {
        for (size_t k = 0; k < small.idx[i].size(); ++k) {
            size_t j     = small.idx[i][k];
            Vec3d  shift = small.shift[i][k];

            // Compute real-space distance using the nn_pos helper
            Vec3d nbr_pos = st_small.nn_pos(i, k);
            Vec3d del;
            del[0] = st_small(i).position[0] - nbr_pos[0];
            del[1] = st_small(i).position[1] - nbr_pos[1];
            del[2] = st_small(i).position[2] - nbr_pos[2];
            double rij_sq = del[0]*del[0] + del[1]*del[1] + del[2]*del[2];

            // Should be within the small cutoff
            REQUIRE(rij_sq < rcut_small_sq * 1.0001);

            // Must also be present in large NN list
            if (!has_nn_entry(st_large, i, j, shift)) {
                err << "Atom " << i << " → nbr " << j
                    << " shift(" << shift[0] << "," << shift[1] << "," << shift[2]
                    << ") dist=" << std::sqrt(rij_sq)
                    << " Å: present in " << rcut_small
                    << " Å list but MISSING from large-cutoff list!\n";
            }
        }
    }
    return err.str();
}

// ---------------------------------------------------------------------------
// Check that nn_pos gives the same position for matching entries in both lists
// ---------------------------------------------------------------------------
static std::string check_positions(const Structure& st_small,
                                   const Structure& st_large)
{
    const auto& small = *st_small.nn;
    std::ostringstream err;
    const double tol = 1e-10;

    for (size_t i = 0; i < st_small.natoms(); ++i) {
        for (size_t k = 0; k < small.idx[i].size(); ++k) {
            size_t j     = small.idx[i][k];
            Vec3d  shift = small.shift[i][k];

            Vec3d pos_small = st_small.nn_pos(i, k);

            // Find matching entry in large NN list
            const auto& large_nn = *st_large.nn;
            for (size_t m = 0; m < large_nn.idx[i].size(); ++m) {
                if (large_nn.idx[i][m] != j) continue;
                const Vec3d& s2 = large_nn.shift[i][m];
                if (std::fabs(s2[0]-shift[0]) > 0.5 ||
                    std::fabs(s2[1]-shift[1]) > 0.5 ||
                    std::fabs(s2[2]-shift[2]) > 0.5) continue;

                Vec3d pos_large = st_large.nn_pos(i, m);
                if (std::fabs(pos_small[0]-pos_large[0]) > tol ||
                    std::fabs(pos_small[1]-pos_large[1]) > tol ||
                    std::fabs(pos_small[2]-pos_large[2]) > tol) {
                    err << "Atom " << i << " → nbr " << j
                        << " shift(" << shift[0] << "," << shift[1] << "," << shift[2]
                        << "): nn_pos differs!\n"
                        << "  small: (" << pos_small[0] << "," << pos_small[1] << "," << pos_small[2] << ")\n"
                        << "  large: (" << pos_large[0] << "," << pos_large[1] << "," << pos_large[2] << ")\n";
                }
                break;
            }
        }
    }
    return err.str();
}

// ===========================================================================
// Tests
// ===========================================================================

TEST_CASE("NNFinder: check_box returns correct values", "[nn][check_box]")
{
    PeriodicTable::initialize();

    SECTION("5.9 Å cell: TRUE for 4.5 Å cutoff, FALSE for 7.0 Å cutoff") {
        Structure st = build_ch4(5.9);
        REQUIRE( NNFinder::check_box(st, 4.5) == true);
        REQUIRE( NNFinder::check_box(st, 7.0) == false);
    }

    SECTION("10.0 Å cell: TRUE for both 4.5 Å and 7.0 Å cutoffs") {
        Structure st = build_ch4(10.0);
        REQUIRE( NNFinder::check_box(st, 4.5) == true);
        REQUIRE( NNFinder::check_box(st, 7.0) == true);
    }
}

TEST_CASE("NNFinder: 5.9 Å cell — 7.0 Å list is superset of 4.5 Å list",
          "[nn][equivalence][cell_5.9]")
{
    // Height 5.9 Å: above 4.5 Å (check_box true), below 7.0 Å (false).
    Structure st_small = build_ch4(5.9);
    Structure st_large = build_ch4(5.9);

    NNFinder nnf;
    nnf.calc(st_small, 4.5);
    nnf.calc(st_large, 7.0);

    REQUIRE(st_small.nn != nullptr);
    REQUIRE(st_large.nn != nullptr);

    // Log NN counts for information
    for (size_t i = 0; i < st_small.natoms(); ++i) {
        INFO("Atom " << i << ": 4.5Å nn_size=" << st_small.nn_size(i)
             << "  7.0Å nn_size=" << st_large.nn_size(i));
        REQUIRE(st_large.nn_size(i) >= st_small.nn_size(i));
    }

    std::string err = check_superset(st_small, st_large, 4.5);
    INFO("Superset check:\n" << err);
    REQUIRE(err.empty());

    std::string pos_err = check_positions(st_small, st_large);
    INFO("Position check:\n" << pos_err);
    REQUIRE(pos_err.empty());
}

TEST_CASE("NNFinder: 10.0 Å cell — 7.0 Å list is superset of 4.5 Å list",
          "[nn][equivalence][cell_10.0]")
{
    // Height 10.0 Å: above both cutoffs (check_box true for both).
    Structure st_small = build_ch4(10.0);
    Structure st_large = build_ch4(10.0);

    NNFinder nnf;
    nnf.calc(st_small, 4.5);
    nnf.calc(st_large, 7.0);

    REQUIRE(st_small.nn != nullptr);
    REQUIRE(st_large.nn != nullptr);

    for (size_t i = 0; i < st_small.natoms(); ++i) {
        INFO("Atom " << i << ": 4.5Å nn_size=" << st_small.nn_size(i)
             << "  7.0Å nn_size=" << st_large.nn_size(i));
        REQUIRE(st_large.nn_size(i) >= st_small.nn_size(i));
    }

    std::string err = check_superset(st_small, st_large, 4.5);
    INFO("Superset check:\n" << err);
    REQUIRE(err.empty());

    std::string pos_err = check_positions(st_small, st_large);
    INFO("Position check:\n" << pos_err);
    REQUIRE(pos_err.empty());
}

TEST_CASE("NNFinder: 5.9 Å cell — the 7.0 Å list filtered to 4.5 Å equals the 4.5 Å list",
          "[nn][filter][cell_5.9]")
{
    // The 4.5 Å list and the 7.0 Å list filtered to 4.5 Å (whose search
    // reaches images beyond the nearest cells, the height being 5.9 Å)
    // must hold exactly the same pairs within 4.5 Å.
    Structure st_small = build_ch4(5.9);
    Structure st_large = build_ch4(5.9);

    NNFinder nnf;
    nnf.calc(st_small, 4.5);
    nnf.calc(st_large, 7.0);

    // Every pair of the 4.5 Å list is in the 7.0 Å list.
    std::string err = check_superset(st_small, st_large, 4.5);
    INFO(err);
    REQUIRE(err.empty());

    // The reverse: every pair of the 7.0 Å list closer than 4.5 Å is in
    // the 4.5 Å list (a filtered view of the 7.0 Å list by distance).
    const double rcut_small_sq = 4.5 * 4.5;
    const auto& large_nn = *st_large.nn;
    std::ostringstream err2;
    for (size_t i = 0; i < st_large.natoms(); ++i) {
        for (size_t m = 0; m < large_nn.idx[i].size(); ++m) {
            Vec3d pos_nbr = st_large.nn_pos(i, m);
            const auto& pos_i = st_large(i).position;
            double del0 = pos_i[0]-pos_nbr[0];
            double del1 = pos_i[1]-pos_nbr[1];
            double del2 = pos_i[2]-pos_nbr[2];
            double rij_sq = del0*del0 + del1*del1 + del2*del2;
            if (rij_sq >= rcut_small_sq) continue;  // beyond small cutoff, skip

            size_t j    = large_nn.idx[i][m];
            Vec3d shift = large_nn.shift[i][m];
            if (!has_nn_entry(st_small, i, j, shift)) {
                err2 << "Atom " << i << " → nbr " << j
                     << " shift(" << shift[0] << "," << shift[1] << "," << shift[2]
                     << ") dist=" << std::sqrt(rij_sq)
                     << " Å: in the 7.0 Å list but MISSING from the 4.5 Å list!\n";
            }
        }
    }
    INFO(err2.str());
    REQUIRE(err2.str().empty());
}

// ===========================================================================
// Tests using actual CH4 phase I database (160-atom structures, 9.96 Å cell).
// These test that the NN superset property holds for real, dense structures
// whose cell height exceeds both the 4.5 Å and the 7.0 Å cutoff.
// ===========================================================================

// The CH4 phase I validation database used during the HPO run; the dataset
// is not redistributed with this repository. The TADAH_TEST_PHASEI_DB
// environment variable names it; unset, the test cases look for
// ch4_phasei_S_U67.tadahdb in the working directory and report themselves
// as skipped when it is not there (TADAH_SKIP: CTest lists them as Skipped,
// not Passed). A database named by TADAH_TEST_PHASEI_DB that cannot be
// opened is a failure: the variable asks for the checks to run.
struct PhaseIDb {
    std::string path;
    bool from_env = false;   // named by TADAH_TEST_PHASEI_DB
};

static PhaseIDb phasei_db() {
    const char* env = std::getenv("TADAH_TEST_PHASEI_DB");
    if (env && *env) return {env, true};
    return {"ch4_phasei_S_U67.tadahdb", false};
}

TEST_CASE("NNFinder: real CH4 phase-I structure — 7.0 Å list is superset of 4.5 Å list",
          "[nn][equivalence][real_ch4_phasei]")
{
    // Load first structure from the database
    const PhaseIDb db = phasei_db();
    std::ifstream ifs(db.path);
    if (!ifs.is_open()) {
        if (db.from_env)
            FAIL("TADAH_TEST_PHASEI_DB=" << db.path << " cannot be opened");
        TADAH_SKIP("CH4 phase-I database not found at " << db.path
                   << " (set TADAH_TEST_PHASEI_DB)");
    }

    PeriodicTable::initialize();
    Structure st_small, st_large;
    st_small.read(ifs);
    ifs.seekg(0);
    st_large.read(ifs);
    ifs.close();

    REQUIRE(st_small.natoms() > 0);

    NNFinder nnf;
    nnf.calc(st_small, 4.5);
    nnf.calc(st_large, 7.0);

    REQUIRE(st_small.nn != nullptr);
    REQUIRE(st_large.nn != nullptr);

    INFO("natoms=" << st_small.natoms()
         << "  cell[0]=" << st_small.cell(0,0));

    // Check_box should be true for both (cell ~9.96 Å > 7.0 Å)
    REQUIRE(NNFinder::check_box(st_small, 4.5) == true);
    REQUIRE(NNFinder::check_box(st_large, 7.0) == true);

    for (size_t i = 0; i < st_small.natoms(); ++i) {
        INFO("Atom " << i << ": 4.5Å nn_size=" << st_small.nn_size(i)
             << "  7.0Å nn_size=" << st_large.nn_size(i));
        REQUIRE(st_large.nn_size(i) >= st_small.nn_size(i));
    }

    std::string err = check_superset(st_small, st_large, 4.5);
    INFO("Superset check:\n" << err);
    REQUIRE(err.empty());

    std::string pos_err = check_positions(st_small, st_large);
    INFO("Position check:\n" << pos_err);
    REQUIRE(pos_err.empty());
}

TEST_CASE("NNFinder: real CH4 phase-I structure — reverse: 4.5 Å list contains all pairs within 4.5 Å from 7.0 Å list",
          "[nn][equivalence][real_ch4_phasei]")
{
    const PhaseIDb db = phasei_db();
    std::ifstream ifs(db.path);
    if (!ifs.is_open()) {
        if (db.from_env)
            FAIL("TADAH_TEST_PHASEI_DB=" << db.path << " cannot be opened");
        TADAH_SKIP("CH4 phase-I database not found at " << db.path
                   << " (set TADAH_TEST_PHASEI_DB)");
    }

    PeriodicTable::initialize();
    Structure st_small, st_large;
    st_small.read(ifs);
    ifs.seekg(0);
    st_large.read(ifs);
    ifs.close();

    REQUIRE(st_small.natoms() > 0);

    NNFinder nnf;
    nnf.calc(st_small, 4.5);
    nnf.calc(st_large, 7.0);

    const double rcut_small_sq = 4.5 * 4.5;
    const auto& large_nn = *st_large.nn;
    std::ostringstream err;
    for (size_t i = 0; i < st_large.natoms(); ++i) {
        for (size_t m = 0; m < large_nn.idx[i].size(); ++m) {
            Vec3d pos_nbr = st_large.nn_pos(i, m);
            const auto& pos_i = st_large(i).position;
            double del0 = pos_i[0]-pos_nbr[0];
            double del1 = pos_i[1]-pos_nbr[1];
            double del2 = pos_i[2]-pos_nbr[2];
            double rij_sq = del0*del0 + del1*del1 + del2*del2;
            if (rij_sq >= rcut_small_sq) continue;

            size_t j    = large_nn.idx[i][m];
            Vec3d shift = large_nn.shift[i][m];
            if (!has_nn_entry(st_small, i, j, shift)) {
                err << "Atom " << i << " → nbr " << j
                    << " shift(" << shift[0] << "," << shift[1] << "," << shift[2]
                    << ") dist=" << std::sqrt(rij_sq)
                    << " Å: in 7.0Å list but MISSING from 4.5Å list!\n";
            }
        }
    }
    INFO(err.str());
    REQUIRE(err.str().empty());
}
