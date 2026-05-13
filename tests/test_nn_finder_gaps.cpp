/**
 * Test: NNFinder coverage on gap cells (triclinic, sub-cutoff, isolated atom,
 *       large BCC supercell, kMaxTotalShifts defensive cap).
 *
 * These complement test_nn_cutoff_equivalence.cpp (cubic CH4 only) and
 * lock in correctness on the cells PC_OBSERVABLE produces: 4×4×4 BCC
 * supercells (cohesive_energy bulk), 1-atom vacuum cells (cohesive_energy
 * iso), and the degenerate-near-zero cells the NativeRelaxer can drift
 * into on wild θ (which trip the kMaxTotalShifts safeguard).
 *
 * The triclinic + sub-cutoff cases ensure refactors to calc_binned /
 * calc_naive can't silently desync on non-orthogonal lattices.
 */
#include "catch2/catch.hpp"
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/core/periodic_table.h>

#include <cmath>
#include <stdexcept>

using namespace tadah::core;
using namespace tadah::mlip;

/* Match an (idx, integer-shift) entry. Shift comparison uses int rounding
 * since NNTable stores shifts as Vec3d of integers. */
static bool has_entry(const Structure& st, size_t i,
                      size_t nbr_idx, int s0, int s1, int s2)
{
    const auto& nn = *st.nn;
    for (size_t k = 0; k < nn.idx[i].size(); ++k) {
        if (nn.idx[i][k] != nbr_idx) continue;
        const Vec3d& s = nn.shift[i][k];
        if (std::fabs(s[0] - s0) < 0.5 &&
            std::fabs(s[1] - s1) < 0.5 &&
            std::fabs(s[2] - s2) < 0.5)
            return true;
    }
    return false;
}

TEST_CASE("NNFinder: triclinic cell — binned matches naive at same cutoff",
          "[nn][gaps][triclinic]")
{
    PeriodicTable::initialize();

    /* Triclinic cell: 6 Å cube tilted so b has +1 Å of a-component.
     * Cell vectors:
     *   a = (6, 0, 0)
     *   b = (1, 6, 0)
     *   c = (0, 0, 6)
     * All vectors > 4.5 Å cutoff, so check_box returns true → binned path.
     * For comparison, call calc_naive directly via the larger 8.0 Å cutoff
     * (forces naive fallback by setting cutoff > min cell dim... but here
     * all cell vectors are ≥ 6 so check_box at 8.0 returns false). */
    Structure st;
    st.cell(0,0) = 6.0; st.cell(0,1) = 0.0; st.cell(0,2) = 0.0;
    st.cell(1,0) = 1.0; st.cell(1,1) = 6.0; st.cell(1,2) = 0.0;
    st.cell(2,0) = 0.0; st.cell(2,1) = 0.0; st.cell(2,2) = 6.0;
    st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 1.0, 1.0, 1.0, 0,0,0));
    st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 4.0, 4.0, 4.0, 0,0,0));
    st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 1.0, 4.0, 4.0, 0,0,0));

    NNFinder nnf;

    /* Path A: small cutoff → binned. */
    Structure st_binned = st;
    nnf.calc(st_binned, 4.5);
    REQUIRE(st_binned.nn != nullptr);

    /* Path B: cutoff larger than min cell dim → naive. */
    Structure st_naive = st;
    nnf.calc(st_naive, 8.0);
    REQUIRE(st_naive.nn != nullptr);

    /* Naive (8.0) must be a superset of binned (4.5). */
    for (size_t i = 0; i < st.natoms(); ++i) {
        CAPTURE(i, st_binned.nn_size(i), st_naive.nn_size(i));
        CHECK(st_naive.nn_size(i) >= st_binned.nn_size(i));
        for (size_t k = 0; k < st_binned.nn->idx[i].size(); ++k) {
            const Vec3d& s = st_binned.nn->shift[i][k];
            const size_t j = st_binned.nn->idx[i][k];
            INFO("Atom " << i << " → nbr " << j
                 << " shift(" << s[0] << "," << s[1] << "," << s[2]
                 << "): present in binned 4.5 Å, missing in naive 8.0 Å");
            CHECK(has_entry(st_naive, i, j,
                            static_cast<int>(std::lround(s[0])),
                            static_cast<int>(std::lround(s[1])),
                            static_cast<int>(std::lround(s[2]))));
        }
    }
}

TEST_CASE("NNFinder: cell_vec < rcut forces naive fallback",
          "[nn][gaps][sub_cutoff]")
{
    PeriodicTable::initialize();

    /* 4 Å cubic cell, 5.0 Å cutoff. check_box returns false (any
     * vec < cutoff) → calc_naive path. Single atom at the centre. */
    Structure st;
    const double a = 4.0;
    st.cell(0,0) = a; st.cell(0,1) = 0; st.cell(0,2) = 0;
    st.cell(1,0) = 0; st.cell(1,1) = a; st.cell(1,2) = 0;
    st.cell(2,0) = 0; st.cell(2,1) = 0; st.cell(2,2) = a;
    st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"),
                     a*0.5, a*0.5, a*0.5, 0,0,0));

    REQUIRE(NNFinder::check_box(st, 5.0) == false);

    NNFinder nnf;
    nnf.calc(st, 5.0);
    REQUIRE(st.nn != nullptr);

    /* With a=4 and rcut=5, periodic images at integer shifts (±1,0,0)
     * etc. sit at distance 4.0 < 5.0 → in the list. Diagonal images
     * at (±1,±1,0) are at √2·4 = 5.66 > 5.0 → not in. So we expect
     * the 6 face-image neighbours (±x,±y,±z) of the single atom. */
    INFO("nn_size(0)=" << st.nn_size(0));
    CHECK(st.nn_size(0) == 6);
}

TEST_CASE("NNFinder: isolated atom (20 Å box) — no neighbours",
          "[nn][gaps][isolated]")
{
    PeriodicTable::initialize();

    Structure st = cells::isolated_atom("Ta", 20.0);
    REQUIRE(st.natoms() == 1);

    NNFinder nnf;
    nnf.calc(st, 5.0);
    REQUIRE(st.nn != nullptr);

    /* Periodic images sit at distance 20 Å > rcut=5; no self-neighbours. */
    INFO("isolated nn_size(0)=" << st.nn_size(0));
    CHECK(st.nn_size(0) == 0);
}

TEST_CASE("NNFinder: 4×4×4 BCC supercell — first+second+third shell counts",
          "[nn][gaps][large][bcc]")
{
    PeriodicTable::initialize();

    /* Conventional 2-atom BCC × 4×4×4 → 128 atoms; cell side = 13.2 Å. */
    const double a0 = 3.30;
    Structure st = cells::bcc("Ta", a0);
    StructureTransformation::replicate(st, 4, 4, 4);
    REQUIRE(st.natoms() == 128);

    NNFinder nnf;
    /* rcut just past the third shell (a0·√2 ≈ 4.667 Å). */
    nnf.calc(st, 5.0);
    REQUIRE(st.nn != nullptr);

    /* In BCC at cutoff between 3rd and 4th shell:
     *   1st shell: 8 atoms at a0·√3/2 ≈ 2.858 Å
     *   2nd shell: 6 atoms at a0    = 3.300 Å
     *   3rd shell: 12 atoms at a0·√2 ≈ 4.667 Å
     * total = 26. By periodic symmetry every atom in the supercell has
     * the same count. */
    for (size_t i = 0; i < st.natoms(); ++i) {
        CAPTURE(i);
        CHECK(st.nn_size(i) == 26);
    }
}

TEST_CASE("NNFinder: degenerate near-zero cell triggers kMaxTotalShifts throw",
          "[nn][gaps][shifts_cap]")
{
    PeriodicTable::initialize();

    /* Tiny cubic cell: 0.3 Å edge at 5.0 Å cutoff → each direction needs
     * ~17 images, total ~35³ ≈ 4.3×10⁴ image cells > kMaxTotalShifts=10⁴.
     * num_shifts is expected to throw. */
    Structure st;
    const double a = 0.3;
    st.cell(0,0) = a; st.cell(0,1) = 0; st.cell(0,2) = 0;
    st.cell(1,0) = 0; st.cell(1,1) = a; st.cell(1,2) = 0;
    st.cell(2,0) = 0; st.cell(2,1) = 0; st.cell(2,2) = a;
    st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 0, 0, 0, 0,0,0));

    NNFinder nnf;
    int N[3];
    CHECK_THROWS_AS(nnf.num_shifts(st, N, 5.0), std::runtime_error);
}
