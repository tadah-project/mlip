/**
 * Test: nn_cache::is_fresh + capture (skin-based NN-list freshness).
 *
 * Mirrors LAMMPS Neighbor::check_distance (neighbor.cpp:2398-2438): the
 * NN list, built at cutoff = rcut + skin, is valid until any atom has
 * moved more than (skin/2 − two-corner shrink) since the last build.
 *
 * The cache is the dominant lever for PC_OBSERVABLE wall-time parity
 * with PC_LAMMPS — without it, NativeRelaxer rebuilds NN on every eval
 * (FD probe, LS trial, force refresh). With it, the long-lived
 * st_inout's NN is reused across iterations as long as cumulative atom
 * + cell drift stay under skin/2.
 */
#include "catch2/catch.hpp"
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/nn_cache.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_transformations.h>

using namespace tadah::core;
using namespace tadah::mlip;

/* Helper: build + capture initial cache state for st (full rebuild). */
static void build_and_capture(Structure& st, double cutoff,
                              nn_cache::State& cache)
{
    NNFinder nnf;
    nnf.calc(st, cutoff);
    nn_cache::capture(st, cache);
}

TEST_CASE("nn_cache: empty cache is never fresh", "[nn_cache][gaps]")
{
    PeriodicTable::initialize();
    Structure st = cells::bcc("Ta", 3.30);
    StructureTransformation::replicate(st, 2, 2, 2);

    nn_cache::State cache;
    cache.skin = 2.0;
    CHECK(nn_cache::is_fresh(st, cache) == false);
}

TEST_CASE("nn_cache: zero displacement → fresh (cache hit)",
          "[nn_cache][trivial]")
{
    PeriodicTable::initialize();
    Structure st = cells::bcc("Ta", 3.30);
    StructureTransformation::replicate(st, 2, 2, 2);

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 7.0 /* rcut+skin */, cache);

    /* Same Structure, no change → fresh. st.nn after is_fresh equals
     * the cached shared_ptr (i.e. exactly what was just captured). */
    auto orig_nn = st.nn;
    st.nn.reset();  /* simulate caller not having st.nn yet */
    REQUIRE(nn_cache::is_fresh(st, cache));
    CHECK(st.nn.get() == cache.cached_nn.get());
    CHECK(st.nn.get() == orig_nn.get());
}

TEST_CASE("nn_cache: small atom displacement (< skin/2) stays fresh",
          "[nn_cache][skin]")
{
    PeriodicTable::initialize();
    Structure st = cells::bcc("Ta", 3.30);
    StructureTransformation::replicate(st, 2, 2, 2);

    nn_cache::State cache;
    cache.skin = 2.0;  /* skin/2 = 1.0 Å */
    build_and_capture(st, 7.0, cache);

    /* Displace atom 0 by (0.3, 0.3, 0.3) Å → |dx| ≈ 0.52 Å < skin/2 = 1.0. */
    st.atoms[0].position[0] += 0.3;
    st.atoms[0].position[1] += 0.3;
    st.atoms[0].position[2] += 0.3;

    CHECK(nn_cache::is_fresh(st, cache) == true);
}

TEST_CASE("nn_cache: large atom displacement (> skin/2) invalidates",
          "[nn_cache][skin]")
{
    PeriodicTable::initialize();
    Structure st = cells::bcc("Ta", 3.30);
    StructureTransformation::replicate(st, 2, 2, 2);

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 7.0, cache);

    /* Displace atom 0 by 1.2 Å in x — exceeds skin/2 = 1.0 Å. */
    st.atoms[0].position[0] += 1.2;

    CHECK(nn_cache::is_fresh(st, cache) == false);
}

TEST_CASE("nn_cache: box expansion erodes effective skin (corner shrink)",
          "[nn_cache][skin][box]")
{
    PeriodicTable::initialize();
    Structure st = cells::bcc("Ta", 3.30);
    StructureTransformation::replicate(st, 4, 4, 4);   /* cell side ≈ 13.2 Å */

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 7.0, cache);

    SECTION("tiny scale change (Δ=0.001) — cache holds") {
        const double s = 1.001;
        for (auto& a : st.atoms) {
            a.position[0] *= s; a.position[1] *= s; a.position[2] *= s;
        }
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                st.cell(r, c) *= s;
        CHECK(nn_cache::is_fresh(st, cache) == true);
    }

    SECTION("large scale change (Δ=0.1) — corner shrink kills the cache") {
        const double s = 1.1;  /* corner shift ~ 0.1 × |corner|; |corner| ≈ 22.9 → ~2.3 Å > skin/2 */
        for (auto& a : st.atoms) {
            a.position[0] *= s; a.position[1] *= s; a.position[2] *= s;
        }
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                st.cell(r, c) *= s;
        CHECK(nn_cache::is_fresh(st, cache) == false);
    }
}

TEST_CASE("nn_cache: atom-count change invalidates",
          "[nn_cache][topology]")
{
    PeriodicTable::initialize();
    Structure st = cells::bcc("Ta", 3.30);
    StructureTransformation::replicate(st, 2, 2, 2);

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 7.0, cache);

    REQUIRE(nn_cache::is_fresh(st, cache));

    /* Remove an atom — natoms mismatch with the captured xref. */
    Structure st_smaller = st;
    st_smaller.atoms.pop_back();
    CHECK(nn_cache::is_fresh(st_smaller, cache) == false);
}
