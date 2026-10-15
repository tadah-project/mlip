/**
 * Test: nn_cache::is_fresh + capture (skin-based NN-list freshness).
 *
 * The NN list is built at cutoff R = rcut + skin. It stays valid for the
 * current positions and cell while every pair closer than rc = R - skin now
 * was closer than R when the list was built. With A the current cell rows,
 * A0 the cached ones, M = A^-1 A0 and e = max_i |r_i M - r0_i| (raw
 * positions), a pair at distance d now was at most d*|M| + 2 e away then,
 * so the list is kept iff (rc + rounding) * B + 2 e + rounding <= R with
 * B = min(sqrt(|M|_1 |M|_inf), 1 + |M - I|_F) >= |M|_2. For a fixed cell
 * (M = I) this is LAMMPS' check_distance (no atom moved more than skin/2).
 * The rule holds for any number of periodic images (thin cells) and for
 * atoms outside the cell. On a hit the neighbour displacements are those
 * of the CURRENT cell: the cached table itself when the cell is bitwise
 * unchanged, otherwise a copy with the displacements recomputed.
 *
 * The cache is the dominant lever for PC_OBSERVABLE wall-time parity
 * with PC_LAMMPS — without it, NativeRelaxer rebuilds NN on every eval
 * (FD probe, LS trial, force refresh).
 */
#include "catch2/catch.hpp"
#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_cache.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/trainer.h>

#include "nn_oracle.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

/* n x n x n supercell of the 2-atom cubic bcc Ta cell, built with the
 * test-local loop of nn_oracle (not StructureTransformation::replicate). */
Structure bcc_supercell(double a, int n)
{
    PeriodicTable::initialize();
    Structure prim;
    prim.label = "bcc Ta";
    prim.cell.load(a, 0.0, 0.0,
                   0.0, a, 0.0,
                   0.0, 0.0, a);
    const auto ta = PeriodicTable::find_by_symbol("Ta");
    prim.add_atom(Atom(ta, 0.0, 0.0, 0.0));
    prim.add_atom(Atom(ta, 0.5 * a, 0.5 * a, 0.5 * a));
    return nn_oracle::supercell(prim, n, n, n);
}

/* Small fixed displacements (<= 0.04 A) so forces are not zero by symmetry. */
void rattle(Structure& st)
{
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        const double t = static_cast<double>(i);
        st.atoms[i].position[0] += 0.04 * std::sin(1.3 * t + 0.2);
        st.atoms[i].position[1] += 0.04 * std::sin(2.1 * t + 0.7);
        st.atoms[i].position[2] += 0.04 * std::sin(0.7 * t + 1.1);
    }
}

/* Affine scaling of the cell rows and the positions. */
void scale_affine(Structure& st, double s)
{
    for (auto& a : st.atoms) {
        a.position[0] *= s; a.position[1] *= s; a.position[2] *= s;
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            st.cell(r, c) *= s;
}

/* A structure whose table is `tbl` (to read a cached table with the
 * oracle helpers). */
Structure with_table(const Structure& st, const NNTablePtr& tbl)
{
    Structure out = st;
    out.nn = tbl;
    return out;
}

/* D2_BP + DM_EAD (Lmax 1) linear model with fixed weights, cutoff rc for
 * both, FORCE and STRESS on. */
Context cache_model_context(double rc)
{
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 73);
    c.add("INIT2B", "true");
    c.add("INITMB", "true");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", 3);
    c.add("TYPE2B", 3);
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("CGRID2B", "2.8 3.6 4.6");
    c.add("SGRID2B", "1.2 1.0 0.8");
    c.add("TYPEMB", "DM_EAD");
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 2);
    c.add("TYPEMB", 2);
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("CGRIDMB", "2.6 3.4");
    c.add("SGRIDMB", "1.0 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B", rc);
    c.add("RCUTMB", rc);
    c.add("DIMER", "false");
    c.add("DIMER", 0);
    c.add("DIMER", "false");
    c.add("FORCE", "true");
    c.add("STRESS", "true");
    c.add("EWEIGHT", 1.0);
    c.add("FWEIGHT", 1.0);
    c.add("SWEIGHT", 1.0);
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    c.add("LAMBDA", 0);
    c.add("OALGO", 1);
    return c;
}

struct FixedModel {
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    Context ctx;
    std::unique_ptr<Trainer> tr;

    explicit FixedModel(double rc) : ctx(cache_model_context(rc))
    {
        std::size_t n_w = 0;
        {
            Context probe_ctx = cache_model_context(rc);
            Trainer probe(probe_ctx, ws);
            n_w = probe.fb->get_phi_cols(probe.context);
        }
        REQUIRE(n_w > 0);
        ctx.add("FIXINDEX", "1-" + std::to_string(n_w));
        for (std::size_t k = 0; k < n_w; ++k)
            ctx.add("FIXWEIGHT",
                    2e-5 * std::cos(1.7 * static_cast<double>(k) + 0.3));
        tr = std::make_unique<Trainer>(ctx, ws);
        tr->provision_fixed_weights();
    }
};

struct Prediction {
    std::vector<aed_type> aed;
    double energy = 0.0;
    std::vector<force_type> forces;
    Matrix3d stress;
};

Prediction predict(Trainer& tr, const Structure& st)
{
    Prediction p;
    const StDescriptors d = tr.dc.calc(st);
    for (std::size_t i = 0; i < st.natoms(); ++i) p.aed.push_back(d.get_aed(i));
    p.energy = tr.model->epredict(d);
    for (std::size_t i = 0; i < st.natoms(); ++i)
        p.forces.push_back(tr.model->fpredict(i, d, st));
    p.stress = tr.model->spredict(d, st);
    return p;
}

}  // namespace

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

TEST_CASE("nn_cache: a cell change keeps the list only while it provably "
          "covers the cutoff",
          "[nn_cache][skin][box]")
{
    /* 4x4x4 bcc (a = 3.3, cell side 13.2 A), list at R = 7, skin 2, so
     * rc = 5. Under an affine scaling s of cell and positions, M = I/s and
     * e = 0: the list is kept iff 5/s <= 7 (up to rounding). The bcc shells
     * at a = 3.3 are 2.86, 3.30, 4.67, 5.47, 5.72, 6.60 and 7.19 A. */
    PeriodicTable::initialize();
    Structure st = bcc_supercell(3.30, 4);
    const Structure st0 = st;   /* the structure the list is built for */

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 7.0, cache);
    const double rc = 5.0;

    auto fresh_and_complete = [&](double s) {
        scale_affine(st, s);
        st.nn.reset();
        const bool fresh = nn_cache::is_fresh(st, cache);
        CHECK(fresh);
        if (!fresh) return;
        REQUIRE(st.nn);
        /* Complete: the pairs a consumer at rc takes from the kept list
         * are exactly the pairs closer than rc in the scaled structure. */
        const auto ref = nn_oracle::reference(st, rc);
        REQUIRE(ref.boundary.empty());
        const auto used = nn_oracle::consumer_filter(st, rc);
        std::size_t missing = 0, extra = 0;
        for (const auto& k : ref.inside) if (!used.count(k)) ++missing;
        for (const auto& k : used) if (!ref.inside.count(k)) ++extra;
        INFO("scale " << s << ": oracle " << ref.inside.size()
             << " pairs inside rc, list gives " << used.size()
             << " (missing " << missing << ", extra " << extra << ")");
        CHECK(missing == 0);
        CHECK(extra == 0);
    };

    SECTION("scale 1.001: fresh and complete") {
        fresh_and_complete(1.001);
    }

    SECTION("scale 1.1: fresh and complete") {
        /* Every pair closer than 5 A after a 10% expansion was closer than
         * 5/1.1 = 4.55 A < 7 A before it. (The old two-corner rule called
         * this stale.) */
        fresh_and_complete(1.1);
    }

    SECTION("scale 0.69: stale, a pair inside rc is missing from the list") {
        scale_affine(st, 0.69);
        CHECK(nn_cache::is_fresh(st, cache) == false);

        /* The reason: the 7.19 A shell of the build structure is at
         * 4.96 A < rc now, and the list (R = 7) does not hold it. */
        const auto ref = nn_oracle::reference(st, rc);
        const auto L0 = nn_oracle::lattice(st0);
        const auto cached = nn_oracle::table_set(with_table(st0, cache.cached_nn));
        bool witness = false;
        long double witness_d0 = 0, witness_d = 0;
        std::size_t absent = 0;
        const auto L = nn_oracle::lattice(st);
        for (const auto& k : ref.inside) {
            if (cached.count(k)) continue;
            ++absent;
            const long double d0 = nn_oracle::norm(nn_oracle::pair_vector(
                L0, st0, std::get<0>(k), std::get<1>(k), std::get<2>(k),
                std::get<3>(k), std::get<4>(k)));
            if (!witness && d0 >= 7.0L) {
                witness = true;
                witness_d0 = d0;
                witness_d = nn_oracle::norm(nn_oracle::pair_vector(
                    L, st, std::get<0>(k), std::get<1>(k), std::get<2>(k),
                    std::get<3>(k), std::get<4>(k)));
            }
        }
        INFO(absent << " pairs inside rc are absent from the list; witness d = "
             << static_cast<double>(witness_d) << " A now, d0 = "
             << static_cast<double>(witness_d0) << " A at the build");
        CHECK(witness);
        CHECK(witness_d < rc);
        CHECK(absent > 0);
    }
}

TEST_CASE("nn_cache: a hit returns the image displacements of the current "
          "cell",
          "[nn_cache][disp]")
{
    /* A 0.1% affine expansion keeps the list (every pair closer than 5.5 A
     * now was closer than 5.5/1.001 A < 7.5 A before). The table's entries
     * are still valid, but its displacements disp = s.A must be those of
     * the scaled cell: a neighbour image is r_j + s.A_current. The cached
     * displacements belong to the old cell, which puts every image up to
     * 0.001 |s.A| (13 mA) off and breaks the descriptors. */
    PeriodicTable::initialize();
    const double R = 7.5, skin = 2.0, rc = R - skin;
    Structure st = bcc_supercell(3.30, 4);
    rattle(st);

    nn_cache::State cache;
    cache.skin = skin;
    build_and_capture(st, R, cache);

    scale_affine(st, 1.001);
    st.nn.reset();
    REQUIRE(nn_cache::is_fresh(st, cache));
    REQUIRE(st.nn);

    /* nn_pos(i, k) == r_j + s0 a0 + s1 a1 + s2 a2 with the current rows. */
    std::size_t n_entries = 0, n_bad = 0;
    double max_err = 0.0;
    const auto& T = *st.nn;
    for (std::size_t i = 0; i < T.idx.size(); ++i)
        for (std::size_t k = 0; k < T.idx[i].size(); ++k) {
            ++n_entries;
            const std::size_t j = T.idx[i][k];
            const auto& s = T.shift[i][k];
            const auto p = st.nn_pos(i, k);
            for (int c = 0; c < 3; ++c) {
                const double want = st.atoms[j].position[c] +
                                    s[0] * st.cell(0, c) +
                                    s[1] * st.cell(1, c) +
                                    s[2] * st.cell(2, c);
                const double err = std::fabs(p[c] - want);
                max_err = std::max(max_err, err);
                if (err > 1e-12 * (1.0 + std::fabs(want))) { ++n_bad; break; }
            }
        }
    INFO(n_bad << " of " << n_entries
         << " neighbour images differ from r_j + s.A (max error "
         << max_err << " A)");
    REQUIRE(n_entries > 0);
    CHECK(n_bad == 0);

    /* The same descriptors, energy, forces and stress as a list built from
     * scratch at R on the scaled structure (both lists hold every pair
     * closer than rc; they may differ beyond it, where the model ignores
     * them). */
    FixedModel fm(rc);
    Structure fresh = st;
    fresh.nn.reset();
    NNFinder().calc(fresh, R);
    const Prediction pc = predict(*fm.tr, st);
    const Prediction pf = predict(*fm.tr, fresh);

    double aed_diff = 0.0, aed_scale = 0.0;
    for (std::size_t i = 0; i < st.natoms(); ++i)
        for (std::size_t k = 0; k < static_cast<std::size_t>(pf.aed[i].size()); ++k) {
            aed_diff = std::max(aed_diff, std::fabs(pc.aed[i][k] - pf.aed[i][k]));
            aed_scale = std::max(aed_scale, std::fabs(pf.aed[i][k]));
        }
    double f_diff = 0.0, f_scale = 0.0;
    for (std::size_t i = 0; i < st.natoms(); ++i)
        for (int c = 0; c < 3; ++c) {
            f_diff = std::max(f_diff, std::fabs(pc.forces[i][c] - pf.forces[i][c]));
            f_scale = std::max(f_scale, std::fabs(pf.forces[i][c]));
        }
    double s_diff = 0.0, s_scale = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            s_diff = std::max(s_diff, std::fabs(pc.stress(a, b) - pf.stress(a, b)));
            s_scale = std::max(s_scale, std::fabs(pf.stress(a, b)));
        }
    INFO("aed: max diff " << aed_diff << " (scale " << aed_scale << ")");
    INFO("energy: cached " << pc.energy << ", fresh " << pf.energy);
    INFO("forces: max diff " << f_diff << " eV/A (scale " << f_scale << ")");
    INFO("stress: max diff " << s_diff << " (scale " << s_scale << ")");
    REQUIRE(aed_scale > 0.0);
    REQUIRE(f_scale > 1e-3);
    REQUIRE(s_scale > 0.0);
    CHECK(aed_diff <= 1e-12 * aed_scale);
    CHECK(pc.energy == Approx(pf.energy).epsilon(1e-12));
    CHECK(f_diff <= 1e-12 * f_scale);
    CHECK(s_diff <= 1e-12 * s_scale);
}

TEST_CASE("nn_cache: the same pointer is kept while the cell is bitwise "
          "unchanged",
          "[nn_cache][pointer]")
{
    /* The displacements depend only on the shifts and the cell, so with an
     * unchanged cell the cached table is exact as it is: no copy. */
    PeriodicTable::initialize();
    Structure st = bcc_supercell(3.30, 3);

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 7.0, cache);

    st.atoms[5].position[0] += 0.10;
    st.atoms[5].position[1] -= 0.05;
    st.nn.reset();
    REQUIRE(nn_cache::is_fresh(st, cache));
    CHECK(st.nn.get() == cache.cached_nn.get());
}

TEST_CASE("nn_cache: a thin cell whose images move further than the box "
          "corners is stale",
          "[nn_cache][thin]")
{
    /* Simple cubic a = 2 with one atom at the origin, list at R = 12, skin 2
     * (rc = 10). Compressing a1 from 2 to 1.55 moves the box corners by
     * 0.45 A only, and the atom not at all, but the images (0, +-6, 0) come
     * from 12 A to 9.3 A: the list (built at 12) misses pairs inside rc.
     * The rule: M = diag(1, 2/1.55, 1), B = 1.29, 10 B = 12.9 > 12. */
    PeriodicTable::initialize();
    Structure st;
    st.label = "simple cubic";
    st.cell.load(2.0, 0.0, 0.0,
                 0.0, 2.0, 0.0,
                 0.0, 0.0, 2.0);
    st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), 0.0, 0.0, 0.0));

    nn_cache::State cache;
    cache.skin = 2.0;
    build_and_capture(st, 12.0, cache);
    const double rc = 12.0 - cache.skin;

    st.cell(1, 1) = 1.55;
    CHECK(nn_cache::is_fresh(st, cache) == false);

    /* The reason, from the oracle: pairs inside rc in the compressed cell
     * that the cached list does not hold. */
    const auto ref = nn_oracle::reference(st, rc);
    const auto cached = nn_oracle::table_set(with_table(st, cache.cached_nn));
    std::size_t missing = 0;
    for (const auto& k : ref.inside) if (!cached.count(k)) ++missing;
    INFO("oracle " << ref.inside.size() << " pairs inside rc; " << missing
         << " of them are absent from the cached list");
    CHECK(missing > 0);
}

TEST_CASE("nn_cache: a list whose cutoff does not exceed the skin is refused",
          "[nn_cache][skin]")
{
    /* rc = R - skin is the cutoff the list serves; a list at R <= skin
     * serves none, which is a caller error (the list must be built at
     * rcut + skin with rcut > 0). */
    PeriodicTable::initialize();
    Structure st = bcc_supercell(3.30, 2);

    SECTION("R = 1.5 < skin = 2") {
        nn_cache::State cache;
        cache.skin = 2.0;
        build_and_capture(st, 1.5, cache);
        CHECK_THROWS_AS(nn_cache::is_fresh(st, cache), std::logic_error);
    }
    SECTION("R = skin = 2") {
        nn_cache::State cache;
        cache.skin = 2.0;
        build_and_capture(st, 2.0, cache);
        CHECK_THROWS_AS(nn_cache::is_fresh(st, cache), std::logic_error);
    }
    SECTION("R = 2.5 > skin = 2 is a valid list") {
        nn_cache::State cache;
        cache.skin = 2.0;
        build_and_capture(st, 2.5, cache);
        CHECK_NOTHROW(nn_cache::is_fresh(st, cache));
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
