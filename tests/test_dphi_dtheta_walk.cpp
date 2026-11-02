// test_dphi_dtheta_walk.cpp
//
// Layer-1.5 parity test for the structure-level dPhi/dtheta walker
// (MLIP/include/tadah/mlip/analytics/dphi_dtheta_walk.h). The walker
// mirrors descriptors_calc.hpp's per-atom + per-neighbour iteration,
// replacing D2_Base::calc_aed with D2_Base::calc_dDdtheta. The test
// builds a small Structure, runs the walker for a known
// (theta_name, arg_pos), and FD-references against a re-walk that
// uses calc_aed at perturbed theta.
//
// Per-pair calc_dDdtheta itself is exercised by
// MODELS/tests/test_d2_bp_dtheta.cpp at Layer 1; this test catches
// integration bugs (atom iteration, neighbour iteration, fidx alignment,
// pair-symmetry scaling) that the per-pair test cannot reach.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/cutoffs.h>
#include <tadah/models/descriptors/d2/d2_bp.h>
#include <tadah/models/descriptors/dm/dm_ead.h>
#include <tadah/models/descriptors/dm/dm_blip.h>
#include <tadah/models/functions/kernels/kern_rbf.h>

#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using tadah::core::aed_type;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::core::rho_type;
using tadah::core::Vec3d;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::models::Cut_Cos;
using tadah::models::D2_BP;
using tadah::models::DM_Blip;
using tadah::models::DM_EAD;
using tadah::models::Kern_RBF;
using tadah::core::Matrix;

// Borrowed from MODELS/tests/test_d2_bp_dtheta.cpp — builds the
// Context boilerplate D2_BP needs for a single H species. RCUT2BMAX is
// intentionally absent; D2_Base falls back to Cut_Dummy(0) at
// construction time and the test attaches a Cut_Cos via set_fcut.
Context make_bp_context(const std::vector<double>& cgrid,
                        const std::vector<double>& sgrid,
                        double rcut)
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("VERBOSE", 0);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid[i];
        }
        c.add("SGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid[i];
        }
        c.add("CGRID2B", ss.str());
    }
    c.add("RCUT2B", rcut);
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", static_cast<int>(cgrid.size()));
    c.add("TYPE2B", static_cast<int>(sgrid.size()));
    c.add("TYPE2B", "H");
    c.add("TYPE2B", "H");
    return c;
}

// Build a 3-atom H Structure in a large cubic cell so all neighbour
// pairs land within the cutoff and home-cell distances dominate (no
// tricky periodic-image pairs).
Structure make_three_atom_structure() {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    st.add_atom(Atom(Element("H"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 1.6, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.0, 2.1, 0.0, 0.0, 0.0, 0.0));
    return st;
}

// Reference: re-walk the same per-atom + per-neighbour loop the
// dphi_dtheta_walker uses, but call D2_Base::calc_aed(unperturbed) to
// build Phi's energy row at the supplied cgrid/sgrid/rcut. Mirrors
// BF_Linear's calc_phi_energy_row (sum AEDs across atoms).
aed_type build_phi_row_via_calc_aed(const std::vector<double>& cgrid,
                                    const std::vector<double>& sgrid,
                                    double rcut,
                                    const Structure& st)
{
    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    aed_type row(d.size());
    for (std::size_t k = 0; k < static_cast<std::size_t>(row.size()); ++k) row[k] = 0.0;

    aed_type aed(d.size());
    const double rcut_sq = rcut * rcut;

    for (std::size_t i = 0; i < st.natoms(); ++i) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed.size()); ++k) aed[k] = 0.0;
        const Atom& a1 = st(i);
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            const double dx = a1.position[0] - a2pos[0];
            const double dy = a1.position[1] - a2pos[1];
            const double dz = a1.position[2] - a2pos[2];
            const double rij_sq = dx*dx + dy*dy + dz*dz;
            if (rij_sq >= rcut_sq) continue;
            const std::size_t neighIdx = st.nn->idx[i][jj];
            const int Zi = a1.Z;
            const int Zj = st(neighIdx).Z;
            const double rij = std::sqrt(rij_sq);
            d.calc_aed(Zi, Zj, rij, rij_sq, aed, /*scale=*/0.5);
        }
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed.size()); ++k)
            row[k] += aed[k];
    }
    return row;
}

// Reference: build the (3·natoms × n_cols) force-row block via the same
// pipeline DesignMatrix uses: calc_all → axis expansion → the antisymmetric
// (fij - fji) subtraction from dm_bf_linear::calc_phi_force_rows. Output is
// row-major: row(3·a + k, n_cols) → flat index (3*a + k) * n_cols + n.
//
// Mirrors descriptors_calc.hpp:226-238 followed by dm_bf_linear.cpp:30-48.
std::vector<double>
build_phi_force_block_via_calc_all(const std::vector<double>& cgrid,
                                   const std::vector<double>& sgrid,
                                   double rcut,
                                   const Structure& st)
{
    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t n_rows = 3 * st.natoms();
    std::vector<double> block(n_rows * n_cols, 0.0);

    // Per-pair scratch matching st_d.fd[a][jj] shape.
    std::vector<std::vector<tadah::core::fd_type>> fd_store(st.natoms());
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        fd_store[a].reserve(st.nn_size(a));
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
            fd_store[a].emplace_back(n_cols);
        }
    }

    const double rcut_sq = rcut * rcut;
    aed_type aed_scratch(d.size());
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed_scratch.size()); ++k)
            aed_scratch[k] = 0.0;
        const Atom& a1 = st(i);
        const int Zi = a1.Z;
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            const double dx = a1.position[0] - a2pos[0];
            const double dy = a1.position[1] - a2pos[1];
            const double dz = a1.position[2] - a2pos[2];
            const double rij_sq = dx*dx + dy*dy + dz*dz;
            if (rij_sq >= rcut_sq) continue;
            const std::size_t neighIdx = st.nn->idx[i][jj];
            const int Zj = st(neighIdx).Z;
            const double rij = std::sqrt(rij_sq);
            const double rij_inv = 1.0 / rij;

            tadah::core::fd_type& fij = fd_store[i][jj];
            d.calc_all(Zi, Zj, rij, rij_sq, aed_scratch, fij, /*scale=*/0.5);

            // Axis expansion mirroring descriptors_calc.hpp:233-238.
            for (std::size_t n = 0; n < n_cols; ++n) {
                fij(n, 0) *= rij_inv;
                fij(n, 1) = fij(n, 0) * dy;
                fij(n, 2) = fij(n, 0) * dz;
                fij(n, 0) *= dx;
            }
        }
    }

    // Antisymmetric pair subtraction (dm_bf_linear::calc_phi_force_rows).
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
            const std::size_t j   = st.nn->idx[a][jj];
            const std::size_t aa  = st.get_nn_iindex(a, j, jj);
            const tadah::core::fd_type& fij = fd_store[a][jj];
            const tadah::core::fd_type& fji = fd_store[j][aa];
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t n = 0; n < n_cols; ++n) {
                    block[(3 * a + k) * n_cols + n] -= (fij(n, k) - fji(n, k));
                }
            }
        }
    }
    return block;
}

}  // namespace

TEST_CASE("dPhi/dtheta walker: PT init", "[dphi_dtheta_walk]") {
    PeriodicTable::initialize();
}

TEST_CASE("dPhi/dtheta walker matches FD on CGRID2B[1] (3-atom H cluster)",
          "[dphi_dtheta_walk][parity][cgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    Structure st = make_three_atom_structure();
    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(std::move(st));

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();   // BIAS=0 in this test setup
    REQUIRE(n_cols >= 1);

    /* Analytical: walk for theta = CGRID2B[1] (the first cgrid entry,
     * 1-based). */
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_d2(
        db, d, "CGRID2B", /*arg_pos=*/1,
        dphi.data(), n_cols);
    REQUIRE(any);

    /* FD: perturb cgrid[0] by ±h, re-walk via calc_aed, finite-diff
     * each component of the energy row. */
    const double h = 1e-5;
    auto cgrid_p = cgrid;  cgrid_p[0] += h;
    auto cgrid_m = cgrid;  cgrid_m[0] -= h;
    aed_type row_p = build_phi_row_via_calc_aed(cgrid_p, sgrid, rcut,
                                                db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed(cgrid_m, sgrid, rcut,
                                                db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-6 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-9);
    }
}

TEST_CASE("dPhi/dtheta walker matches FD on SGRID2B[2]",
          "[dphi_dtheta_walk][parity][sgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();

    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_d2(
        db, d, "SGRID2B", /*arg_pos=*/2, dphi.data(), n_cols);
    REQUIRE(any);

    const double h = 1e-5;
    auto sgrid_p = sgrid;  sgrid_p[1] += h;
    auto sgrid_m = sgrid;  sgrid_m[1] -= h;
    aed_type row_p = build_phi_row_via_calc_aed(cgrid, sgrid_p, rcut,
                                                db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed(cgrid, sgrid_m, rcut,
                                                db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-6 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-9);
    }
}

// ============================================================================
// Force-row walker — Layer-1.5 parity for the per-D2 force
// walker. Builds the (3·natoms × n_cols) FD reference via
// build_phi_force_block_via_calc_all at θ±h and compares against the
// analytical walker output.
// ============================================================================

TEST_CASE("dPhi/dtheta force walker matches FD on CGRID2B[1] (3-atom H cluster)",
          "[dphi_dtheta_walk][force][parity][cgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t natoms = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;

    /* Analytical: walker output. force_row_start[0] = 0 since this
     * test has only one structure and we put force rows at the top. */
    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "CGRID2B", /*arg_pos=*/1,
        dphi.data(), n_cols, force_row_start);
    REQUIRE(any);

    /* FD: rebuild the force block at θ±h via calc_all + axis expansion +
     * antisymmetric subtraction, then central-diff. */
    const double h = 1e-5;
    auto cgrid_p = cgrid;  cgrid_p[0] += h;
    auto cgrid_m = cgrid;  cgrid_m[0] -= h;
    auto block_p = build_phi_force_block_via_calc_all(cgrid_p, sgrid, rcut,
                                                      db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_all(cgrid_m, sgrid, rcut,
                                                      db.structures[0]);

    for (std::size_t a = 0; a < natoms; ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (std::size_t n = 0; n < n_cols; ++n) {
                const std::size_t idx = (3 * a + k) * n_cols + n;
                const double an = dphi[idx];
                const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
                INFO("atom=" << a << " axis=" << k << " col=" << n
                     << " analytical=" << an << " FD=" << fd);
                const double tol = 1e-6 * std::max({std::abs(an), std::abs(fd), 1e-9});
                CHECK(std::abs(an - fd) <= tol + 1e-9);
            }
        }
    }
}

TEST_CASE("dPhi/dtheta force walker matches FD on SGRID2B[2]",
          "[dphi_dtheta_walk][force][parity][sgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t natoms = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;

    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "SGRID2B", /*arg_pos=*/2,
        dphi.data(), n_cols, force_row_start);
    REQUIRE(any);

    const double h = 1e-5;
    auto sgrid_p = sgrid;  sgrid_p[1] += h;
    auto sgrid_m = sgrid;  sgrid_m[1] -= h;
    auto block_p = build_phi_force_block_via_calc_all(cgrid, sgrid_p, rcut,
                                                      db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_all(cgrid, sgrid_m, rcut,
                                                      db.structures[0]);

    for (std::size_t a = 0; a < natoms; ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (std::size_t n = 0; n < n_cols; ++n) {
                const std::size_t idx = (3 * a + k) * n_cols + n;
                const double an = dphi[idx];
                const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
                INFO("atom=" << a << " axis=" << k << " col=" << n
                     << " analytical=" << an << " FD=" << fd);
                const double tol = 1e-6 * std::max({std::abs(an), std::abs(fd), 1e-9});
                CHECK(std::abs(an - fd) <= tol + 1e-9);
            }
        }
    }
}

TEST_CASE("dPhi/dtheta force walker matches FD on RCUT2B",
          "[dphi_dtheta_walk][force][parity][rcut]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    // Pad NNFinder cutoff by 1e-4 so the rcut±h FD evaluations see the
    // same neighbour set as the analytical pass (same trick the DM-RCUTMB
    // walker test uses).
    NNFinder finder;
    finder.calc(db.structures[0], rcut + 1e-4);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t natoms = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;

    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "RCUT2B", /*arg_pos=*/-1,
        dphi.data(), n_cols, force_row_start);
    REQUIRE(any);

    const double h = 1e-5;
    auto block_p = build_phi_force_block_via_calc_all(cgrid, sgrid, rcut + h,
                                                      db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_all(cgrid, sgrid, rcut - h,
                                                      db.structures[0]);

    for (std::size_t a = 0; a < natoms; ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (std::size_t n = 0; n < n_cols; ++n) {
                const std::size_t idx = (3 * a + k) * n_cols + n;
                const double an = dphi[idx];
                const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
                INFO("atom=" << a << " axis=" << k << " col=" << n
                     << " analytical=" << an << " FD=" << fd);
                const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
                CHECK(std::abs(an - fd) <= tol + 1e-8);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Force walker parity on D2_Blip. The walker is
// descriptor-polymorphic over D2_Base&, so this case exercises the same
// integration code path with a different per-pair closed form. Test
// points keep x_c = eta(r-mu) at least 0.2 away from |x|=1 (C^2 break).
// ----------------------------------------------------------------------

namespace {

Context make_blip_context_mlip(const std::vector<double>& cgrid,
                               const std::vector<double>& sgrid,
                               double rcut)
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("VERBOSE", 0);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid[i];
        }
        c.add("SGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid[i];
        }
        c.add("CGRID2B", ss.str());
    }
    c.add("RCUT2B", rcut);
    c.add("TYPE2B", "D2_Blip");
    c.add("TYPE2B", static_cast<int>(cgrid.size()));
    c.add("TYPE2B", static_cast<int>(sgrid.size()));
    c.add("TYPE2B", "H");
    c.add("TYPE2B", "H");
    return c;
}

// FD reference for D2_Blip: walk calc_all + axis expansion + the
// (fij - fji) antisymmetric subtraction at the given θ.
std::vector<double>
build_blip_phi_force_block_via_calc_all(const std::vector<double>& cgrid,
                                        const std::vector<double>& sgrid,
                                        double rcut,
                                        const Structure& st)
{
    Context c = make_blip_context_mlip(cgrid, sgrid, rcut);
    tadah::models::D2_Blip d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t n_rows = 3 * st.natoms();
    std::vector<double> block(n_rows * n_cols, 0.0);

    std::vector<std::vector<tadah::core::fd_type>> fd_store(st.natoms());
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        fd_store[a].reserve(st.nn_size(a));
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj)
            fd_store[a].emplace_back(n_cols);
    }

    const double rcut_sq = rcut * rcut;
    aed_type aed_scratch(d.size());
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed_scratch.size()); ++k)
            aed_scratch[k] = 0.0;
        const Atom& a1 = st(i);
        const int Zi = a1.Z;
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            const double dx = a1.position[0] - a2pos[0];
            const double dy = a1.position[1] - a2pos[1];
            const double dz = a1.position[2] - a2pos[2];
            const double rij_sq = dx*dx + dy*dy + dz*dz;
            if (rij_sq >= rcut_sq) continue;
            const std::size_t neighIdx = st.nn->idx[i][jj];
            const int Zj = st(neighIdx).Z;
            const double rij = std::sqrt(rij_sq);
            const double rij_inv = 1.0 / rij;

            tadah::core::fd_type& fij = fd_store[i][jj];
            d.calc_all(Zi, Zj, rij, rij_sq, aed_scratch, fij, /*scale=*/0.5);

            for (std::size_t n = 0; n < n_cols; ++n) {
                fij(n, 0) *= rij_inv;
                fij(n, 1) = fij(n, 0) * dy;
                fij(n, 2) = fij(n, 0) * dz;
                fij(n, 0) *= dx;
            }
        }
    }

    for (std::size_t a = 0; a < st.natoms(); ++a) {
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
            const std::size_t j   = st.nn->idx[a][jj];
            const std::size_t aa  = st.get_nn_iindex(a, j, jj);
            const tadah::core::fd_type& fij = fd_store[a][jj];
            const tadah::core::fd_type& fji = fd_store[j][aa];
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t n = 0; n < n_cols; ++n) {
                    block[(3 * a + k) * n_cols + n] -= (fij(n, k) - fji(n, k));
                }
            }
        }
    }
    return block;
}

}  // namespace

TEST_CASE("dPhi/dtheta force walker matches FD on D2_Blip CGRID2B[1] (interior x_c)",
          "[dphi_dtheta_walk][force][parity][cgrid][blip]")
{
    PeriodicTable::initialize();

    // x_c at rij=1.6: eta[0]*(1.6-1.0)=1.5*0.6=+0.9   (interior, margin 0.1 — too close)
    // Use larger margin: cgrid={0.8, 1.5}, sgrid={1.5, 0.8}, expected
    //   x_c[0] = 1.5 * (1.6 - 0.8) = +1.2  (outer cubic, margin 0.2 from 1, 0.8 from 2)
    //   x_c[1] = 0.8 * (1.6 - 1.5) = +0.08 (interior, far from |x|=1)
    // Other r-pairs hit different x_c values; rely on r-spread of the cluster.
    const std::vector<double> cgrid = {0.8, 1.5};
    const std::vector<double> sgrid = {1.5, 0.8};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_blip_context_mlip(cgrid, sgrid, rcut);
    tadah::models::D2_Blip d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t n_force_rows = 3 * db.structures[0].natoms();

    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "CGRID2B", /*arg_pos=*/1,
        dphi.data(), n_cols, force_row_start);
    REQUIRE(any);

    const double h = 1e-5;
    auto cgrid_p = cgrid;  cgrid_p[0] += h;
    auto cgrid_m = cgrid;  cgrid_m[0] -= h;
    auto block_p = build_blip_phi_force_block_via_calc_all(cgrid_p, sgrid, rcut,
                                                           db.structures[0]);
    auto block_m = build_blip_phi_force_block_via_calc_all(cgrid_m, sgrid, rcut,
                                                           db.structures[0]);

    for (std::size_t a = 0; a < db.structures[0].natoms(); ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (std::size_t n = 0; n < n_cols; ++n) {
                const std::size_t idx = (3 * a + k) * n_cols + n;
                const double an = dphi[idx];
                const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
                INFO("atom=" << a << " axis=" << k << " col=" << n
                     << " analytical=" << an << " FD=" << fd);
                const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
                CHECK(std::abs(an - fd) <= tol + 1e-8);
            }
        }
    }
}

TEST_CASE("dPhi/dtheta force walker matches FD on D2_Blip RCUT2B",
          "[dphi_dtheta_walk][force][parity][rcut][blip]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.8, 1.5};
    const std::vector<double> sgrid = {1.5, 0.8};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    // Pad NN cutoff so the rcut±h FD evaluations see the same neighbour
    // set as the analytical pass.
    NNFinder finder;
    finder.calc(db.structures[0], rcut + 1e-4);

    Context c = make_blip_context_mlip(cgrid, sgrid, rcut);
    tadah::models::D2_Blip d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t n_force_rows = 3 * db.structures[0].natoms();

    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "RCUT2B", /*arg_pos=*/-1,
        dphi.data(), n_cols, force_row_start);
    REQUIRE(any);

    const double h = 1e-5;
    auto block_p = build_blip_phi_force_block_via_calc_all(cgrid, sgrid, rcut + h,
                                                           db.structures[0]);
    auto block_m = build_blip_phi_force_block_via_calc_all(cgrid, sgrid, rcut - h,
                                                           db.structures[0]);

    for (std::size_t a = 0; a < db.structures[0].natoms(); ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (std::size_t n = 0; n < n_cols; ++n) {
                const std::size_t idx = (3 * a + k) * n_cols + n;
                const double an = dphi[idx];
                const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
                INFO("atom=" << a << " axis=" << k << " col=" << n
                     << " analytical=" << an << " FD=" << fd);
                const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
                CHECK(std::abs(an - fd) <= tol + 1e-8);
            }
        }
    }
}

TEST_CASE("dPhi/dtheta force walker reports false for unregistered theta",
          "[dphi_dtheta_walk][force][routing]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5};
    const std::vector<double> sgrid = {3.0};
    const double rcut = 2.5;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t n_force_rows = 3 * db.structures[0].natoms();

    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};

    const bool any = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "WATOM", /*arg_pos=*/-1,
        dphi.data(), n_cols, force_row_start);
    CHECK_FALSE(any);
    for (double v : dphi) CHECK(v == 0.0);
}

TEST_CASE("dPhi/dtheta walker reports false for unregistered theta",
          "[dphi_dtheta_walk][routing]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5};
    const std::vector<double> sgrid = {3.0};
    const double rcut = 2.5;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());

    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    std::vector<double> dphi(db.size() * n_cols, 0.0);

    /* "WATOM" is a real config key but D2_BP doesn't own a derivative
     * for it; the walker should report false and leave dphi untouched. */
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_d2(
        db, d, "WATOM", /*arg_pos=*/-1, dphi.data(), n_cols);
    CHECK_FALSE(any);
    for (double v : dphi) CHECK(v == 0.0);
}

// ============================================================================
// DM walker — DM_EAD and DM_Blip many-body parity.
//
// Same structure as the D2 walker tests: build a structure, run the walker
// for a known (theta_name, arg_pos), FD-reference against a fresh pipeline
// run at perturbed theta. The reference function mirrors descriptors_calc.hpp
// (calc_rho per-pair, then dm.calc_aed per-atom) and sums atom AEDs into
// the structure energy row.
//
// Cluster geometry uses a 4-atom Structure in a large cubic cell so that
// the home-cell neighbour pairs dominate (no tricky periodic images).
// Neighbour placement is the same shape as the per-atom Layer-1 cluster
// in test_dm_ead_dtheta.cpp.
// ============================================================================

Context make_ead_context_mlip(const std::vector<double>& cgrid,
                              const std::vector<double>& sgrid,
                              double rcut,
                              int    Lmax = 2)
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1);
    c.add("INITMB", "true");
    c.add("VERBOSE", 0);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid[i];
        }
        c.add("SGRIDMB", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid[i];
        }
        c.add("CGRIDMB", ss.str());
    }
    c.add("RCUTMB", rcut);
    c.add("TYPEMB", "DM_EAD");
    c.add("TYPEMB", Lmax);
    c.add("TYPEMB", static_cast<int>(cgrid.size()));
    c.add("TYPEMB", static_cast<int>(sgrid.size()));
    c.add("TYPEMB", "H");
    c.add("TYPEMB", "H");
    return c;
}

Context make_blip_mb_context(const std::vector<double>& cgrid,
                             const std::vector<double>& sgrid,
                             double rcut,
                             int    Lmax = 2)
{
    Context c = make_ead_context_mlip(cgrid, sgrid, rcut, Lmax);
    /* Patch TYPEMB[0] to DM_Blip. Context::add appends, so replace the
     * label by going through a fresh ctx. */
    Context c2;
    c2.add("ATOM", "H");
    c2.add("WATOM", 1);
    c2.add("INITMB", "true");
    c2.add("VERBOSE", 0);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid[i];
        }
        c2.add("SGRIDMB", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid[i];
        }
        c2.add("CGRIDMB", ss.str());
    }
    c2.add("RCUTMB", rcut);
    c2.add("TYPEMB", "DM_Blip");
    c2.add("TYPEMB", Lmax);
    c2.add("TYPEMB", static_cast<int>(cgrid.size()));
    c2.add("TYPEMB", static_cast<int>(sgrid.size()));
    c2.add("TYPEMB", "H");
    c2.add("TYPEMB", "H");
    return c2;
}

// Build a 4-atom H Structure with neighbours at r ~ {1.6, 1.4, 1.9}.
// Cell is large enough that all NN pairs are home-cell distances.
Structure make_four_atom_structure() {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    st.add_atom(Atom(Element("H"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 1.6, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.0, 1.4, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.0, 0.0, 1.9, 0.0, 0.0, 0.0));
    return st;
}

// Reference: build a structure energy row of Phi via the standard
// pipeline (calc_rho per pair, then dm.calc_aed per atom, sum across
// atoms). Mirrors what BF_Linear's calc_phi_energy_row does for the
// MB block, with no DesignMatrix weighting (so it matches the walker's
// row layout).
template <typename DM>
aed_type build_phi_row_via_calc_aed_dm(const Context& c,
                                       double rcut,
                                       const Structure& st)
{
    DM dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    aed_type row(dm.size());
    for (std::size_t k = 0; k < static_cast<std::size_t>(row.size()); ++k)
        row[k] = 0.0;

    const double rcut_sq = rcut * rcut;
    Vec3d delij;
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        rho_type rhoi;
        dm.init_rhoi(rhoi);
        rhoi.set_zero();

        const Atom& a1 = st(i);
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            delij[0] = a1.position[0] - a2pos[0];
            delij[1] = a1.position[1] - a2pos[1];
            delij[2] = a1.position[2] - a2pos[2];
            const double rij_sq = delij*delij;
            if (rij_sq >= rcut_sq) continue;
            const double rij = std::sqrt(rij_sq);
            dm.calc_rho(1, 1, rij, rij_sq, delij, rhoi, /*scale=*/1.0);
        }

        aed_type aed(dm.size());
        aed.set_zero();
        dm.calc_aed(rhoi, aed);
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed.size()); ++k)
            row[k] += aed[k];
    }
    return row;
}

TEST_CASE("DM walker (DM_EAD) matches FD on CGRIDMB[1]",
          "[dphi_dtheta_walk][dm][parity][cgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.7, 1.5};
    const std::vector<double> sgrid = {2.0, 3.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_ead_context_mlip(cgrid, sgrid, rcut, Lmax);
    DM_EAD dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();   // BIAS=0 in this setup; no D2 arm
    REQUIRE(n_cols >= 1);

    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1, dphi.data(), n_cols);
    REQUIRE(any);

    const double h = 1e-5;
    auto cp = cgrid;  cp[0] += h;
    auto cm = cgrid;  cm[0] -= h;
    Context cp_ctx = make_ead_context_mlip(cp, sgrid, rcut, Lmax);
    Context cm_ctx = make_ead_context_mlip(cm, sgrid, rcut, Lmax);
    aed_type row_p = build_phi_row_via_calc_aed_dm<DM_EAD>(cp_ctx, rcut, db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed_dm<DM_EAD>(cm_ctx, rcut, db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-7);
    }
}

TEST_CASE("DM walker (DM_EAD) matches FD on SGRIDMB[2]",
          "[dphi_dtheta_walk][dm][parity][sgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.7, 1.5};
    const std::vector<double> sgrid = {2.0, 3.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_ead_context_mlip(cgrid, sgrid, rcut, Lmax);
    DM_EAD dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();

    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "SGRIDMB", /*arg_pos=*/2, dphi.data(), n_cols);
    REQUIRE(any);

    const double h = 1e-5;
    auto sp = sgrid;  sp[1] += h;
    auto sm = sgrid;  sm[1] -= h;
    Context sp_ctx = make_ead_context_mlip(cgrid, sp, rcut, Lmax);
    Context sm_ctx = make_ead_context_mlip(cgrid, sm, rcut, Lmax);
    aed_type row_p = build_phi_row_via_calc_aed_dm<DM_EAD>(sp_ctx, rcut, db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed_dm<DM_EAD>(sm_ctx, rcut, db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-7);
    }
}

TEST_CASE("DM walker (DM_EAD) matches FD on RCUTMB",
          "[dphi_dtheta_walk][dm][parity][rcut]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.7, 1.5};
    const std::vector<double> sgrid = {2.0, 3.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut + 1e-4);  /* slight pad for h perturbation */

    Context c = make_ead_context_mlip(cgrid, sgrid, rcut, Lmax);
    DM_EAD dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();

    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "RCUTMB", /*arg_pos=*/-1, dphi.data(), n_cols);
    REQUIRE(any);

    const double h = 1e-5;
    Context cp_ctx = make_ead_context_mlip(cgrid, sgrid, rcut + h, Lmax);
    Context cm_ctx = make_ead_context_mlip(cgrid, sgrid, rcut - h, Lmax);
    aed_type row_p = build_phi_row_via_calc_aed_dm<DM_EAD>(cp_ctx, rcut + h, db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed_dm<DM_EAD>(cm_ctx, rcut - h, db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(a), std::abs(fd), 1e-9});
        /* RCUTMB hits both fc and the L-prefactor; widen the floor. */
        CHECK(std::abs(a - fd) <= tol + 1e-6);
    }
}

TEST_CASE("DM walker (DM_Blip) matches FD on CGRIDMB[1]",
          "[dphi_dtheta_walk][dm][parity][blip]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.7, 1.5};
    const std::vector<double> sgrid = {1.5, 0.8};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_blip_mb_context(cgrid, sgrid, rcut, Lmax);
    DM_Blip dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();

    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1, dphi.data(), n_cols);
    REQUIRE(any);

    const double h = 1e-5;
    auto cp = cgrid;  cp[0] += h;
    auto cm = cgrid;  cm[0] -= h;
    Context cp_ctx = make_blip_mb_context(cp, sgrid, rcut, Lmax);
    Context cm_ctx = make_blip_mb_context(cm, sgrid, rcut, Lmax);
    aed_type row_p = build_phi_row_via_calc_aed_dm<DM_Blip>(cp_ctx, rcut, db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed_dm<DM_Blip>(cm_ctx, rcut, db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-6);
    }
}

/* ----------------------------------------------------------------------
 * DM force-row walker: Layer-1.5 parity for
 * assemble_dphi_train_force_rows_dm.
 *
 * Cluster: same 4-atom H cluster as the energy-walker tests.
 * Descriptor: DM_mBlip (= DM_mEAD<F_Blip>) — ng==1, odim = CEMBFUNC.size.
 *
 * FD reference: rebuild the full (3·natoms × n_cols) force block at
 * θ ± h using the standard descriptors_calc pipeline (per-atom
 * calc_rho + calc_aed, per-pair calc_dXijdri, then the antisymmetric
 * (fij - fji) subtraction from dm_bf_linear::calc_phi_force_rows).
 * ---------------------------------------------------------------------- */

namespace {

using tadah::models::DM_mEAD;
using tadah::models::F_Blip;

/* Off-axis 4-atom H cluster. Same shape as make_four_atom_structure
 * but with neighbours that are non-zero in all 3 Cartesian components,
 * so the per-orbital ρ for L>0 mixed polynomials never collapses to
 * exactly zero. A zero ρ pins x_c = -mu_0*eta_0 onto b''(x_c)'s C^1
 * break in F_Blip-driven embedding theta — FD truncation around that
 * point dominates the analytical-vs-FD discrepancy. */
Structure make_four_atom_off_axis_structure() {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    st.add_atom(Atom(Element("H"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 1.2, 0.7, 0.4, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.3, 1.1, 0.6, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.5, 0.6, 1.3, 0.0, 0.0, 0.0));
    return st;
}

Context make_mblip_mb_context_mlip(double cgrid_val, double sgrid_val,
                                    const std::vector<double>& cembfunc,
                                    const std::vector<double>& sembfunc,
                                    double rcut, int Lmax = 2)
{
    Context c;
    c.add("ATOM",   "H");
    c.add("WATOM",  1);
    c.add("INITMB", "true");
    c.add("VERBOSE", 0);
    c.add("SGRIDMB", sgrid_val);
    c.add("CGRIDMB", cgrid_val);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cembfunc.size(); ++i) {
            if (i) ss << " ";
            ss << cembfunc[i];
        }
        c.add("CEMBFUNC", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sembfunc.size(); ++i) {
            if (i) ss << " ";
            ss << sembfunc[i];
        }
        c.add("SEMBFUNC", ss.str());
    }
    c.add("RCUTMB", rcut);
    /* TYPEMB layout for DM_mEAD<F_Blip>: nparams=5 → DM_Base::get_init_atoms
     * drops the first nparams+1 = 6 elements (label + Lmax + 4 key sizes). */
    c.add("TYPEMB", "DM_mBlip");
    c.add("TYPEMB", Lmax);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", static_cast<int>(cembfunc.size()));
    c.add("TYPEMB", static_cast<int>(sembfunc.size()));
    c.add("TYPEMB", "H");
    c.add("TYPEMB", "H");
    return c;
}

/* Build the (3·natoms × n_cols) force block via the DM standard pipeline
 * (per-atom calc_rho + calc_aed, per-pair calc_dXijdri, antisymmetric
 * (fij - fji) subtraction). Mirrors descriptors_calc.hpp's MB path
 * followed by dm_bf_linear.cpp:30-48. Row-major: (3*a + k) * n_cols + n. */
std::vector<double>
build_phi_force_block_via_calc_dXijdri_dm(const Context& c,
                                          double rcut,
                                          const Structure& st)
{
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    const std::size_t natoms = st.natoms();
    const std::size_t rho_total = dm.rhoi_size() + dm.rhoip_size();

    /* Per-atom rho with F'(rho) tail populated by calc_aed. */
    std::vector<tadah::core::rho_type> rhoi_per_atom(natoms);
    aed_type aed_scratch(n_cols);
    const double rcut_sq = rcut * rcut;
    Vec3d delij;
    for (std::size_t i = 0; i < natoms; ++i) {
        rhoi_per_atom[i].resize(static_cast<long>(rho_total));
        rhoi_per_atom[i].set_zero();
        const Atom& a1 = st(i);
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            delij[0] = a1.position[0] - a2pos[0];
            delij[1] = a1.position[1] - a2pos[1];
            delij[2] = a1.position[2] - a2pos[2];
            const double rij_sq = delij*delij;
            if (rij_sq >= rcut_sq) continue;
            const double rij = std::sqrt(rij_sq);
            dm.calc_rho(1, 1, rij, rij_sq, delij, rhoi_per_atom[i], /*scale=*/1.0);
        }
        aed_scratch.set_zero();
        dm.calc_aed(rhoi_per_atom[i], aed_scratch);
    }

    /* Per-pair fd_ij (axis-expanded). Stored as st_d.fd[a][jj]-style 2D. */
    std::vector<std::vector<tadah::core::fd_type>> fd_store(natoms);
    for (std::size_t a = 0; a < natoms; ++a) {
        fd_store[a].reserve(st.nn_size(a));
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
            fd_store[a].emplace_back(n_cols);
        }
    }

    for (std::size_t i = 0; i < natoms; ++i) {
        const Atom& a1 = st(i);
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            delij[0] = a1.position[0] - a2pos[0];
            delij[1] = a1.position[1] - a2pos[1];
            delij[2] = a1.position[2] - a2pos[2];
            const double rij_sq = delij*delij;
            if (rij_sq >= rcut_sq) continue;
            const double rij = std::sqrt(rij_sq);
            tadah::core::fd_type& fij = fd_store[i][jj];
            for (std::size_t n = 0; n < n_cols; ++n)
                for (std::size_t k = 0; k < 3; ++k)
                    fij(n, k) = 0.0;
            dm.calc_dXijdri(1, 1, rij, rij_sq, delij,
                            rhoi_per_atom[i], fij, /*scale=*/1.0);
        }
    }

    /* Antisymmetric (fij - fji) subtraction into atom i's three rows. */
    std::vector<double> block(3 * natoms * n_cols, 0.0);
    for (std::size_t a = 0; a < natoms; ++a) {
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
            const std::size_t j  = st.nn->idx[a][jj];
            const std::size_t aa = st.get_nn_iindex(a, j, jj);
            const tadah::core::fd_type& fij = fd_store[a][jj];
            const tadah::core::fd_type& fji = fd_store[j][aa];
            for (std::size_t k = 0; k < 3; ++k) {
                for (std::size_t n = 0; n < n_cols; ++n) {
                    block[(3 * a + k) * n_cols + n] -= (fij(n, k) - fji(n, k));
                }
            }
        }
    }
    return block;
}

}  // namespace

TEST_CASE("DM force walker (DM_mBlip) matches FD on CGRIDMB[1]",
          "[dphi_dtheta_walk][dm][force][parity][mblip]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    const std::size_t natoms = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;

    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1,
        dphi.data(), n_cols, force_row_start);
    REQUIRE(any);

    const double h = 1e-5;
    Context cp = make_mblip_mb_context_mlip(cgrid_val + h, sgrid_val,
                                             cembfunc, sembfunc, rcut, Lmax);
    Context cm = make_mblip_mb_context_mlip(cgrid_val - h, sgrid_val,
                                             cembfunc, sembfunc, rcut, Lmax);
    auto block_p = build_phi_force_block_via_calc_dXijdri_dm(cp, rcut,
                                                              db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_dXijdri_dm(cm, rcut,
                                                              db.structures[0]);

    for (std::size_t a = 0; a < natoms; ++a) {
        for (std::size_t k = 0; k < 3; ++k) {
            for (std::size_t n = 0; n < n_cols; ++n) {
                const std::size_t idx = (3 * a + k) * n_cols + n;
                const double an = dphi[idx];
                const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
                INFO("atom=" << a << " axis=" << k << " col=" << n
                     << " analytical=" << an << " FD=" << fd);
                const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
                CHECK(std::abs(an - fd) <= tol + 1e-6);
            }
        }
    }
}

TEST_CASE("DM force walker (DM_mBlip) matches FD on SGRIDMB[1]",
          "[dphi_dtheta_walk][dm][force][parity][mblip]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    const std::size_t natoms = db.structures[0].natoms();

    std::vector<double> dphi(3 * natoms * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    REQUIRE(tadah::mlip::assemble_dphi_train_force_rows_dm(
        db, dm, "SGRIDMB", /*arg_pos=*/1,
        dphi.data(), n_cols, force_row_start));

    const double h = 1e-5;
    Context cp = make_mblip_mb_context_mlip(cgrid_val, sgrid_val + h,
                                             cembfunc, sembfunc, rcut, Lmax);
    Context cm = make_mblip_mb_context_mlip(cgrid_val, sgrid_val - h,
                                             cembfunc, sembfunc, rcut, Lmax);
    auto block_p = build_phi_force_block_via_calc_dXijdri_dm(cp, rcut,
                                                              db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_dXijdri_dm(cm, rcut,
                                                              db.structures[0]);

    for (std::size_t idx = 0; idx < dphi.size(); ++idx) {
        const double an = dphi[idx];
        const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
        INFO("idx=" << idx << " analytical=" << an << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
        CHECK(std::abs(an - fd) <= tol + 1e-6);
    }
}

TEST_CASE("DM force walker (DM_mBlip) matches FD on CEMBFUNC[1]",
          "[dphi_dtheta_walk][dm][force][parity][mblip][embedding]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    const std::size_t natoms = db.structures[0].natoms();

    std::vector<double> dphi(3 * natoms * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    REQUIRE(tadah::mlip::assemble_dphi_train_force_rows_dm(
        db, dm, "CEMBFUNC", /*arg_pos=*/1,
        dphi.data(), n_cols, force_row_start));

    const double h = 1e-5;
    auto cembp = cembfunc; cembp[0] += h;
    auto cembm = cembfunc; cembm[0] -= h;
    Context cp = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                             cembp, sembfunc, rcut, Lmax);
    Context cm = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                             cembm, sembfunc, rcut, Lmax);
    auto block_p = build_phi_force_block_via_calc_dXijdri_dm(cp, rcut,
                                                              db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_dXijdri_dm(cm, rcut,
                                                              db.structures[0]);

    for (std::size_t idx = 0; idx < dphi.size(); ++idx) {
        const double an = dphi[idx];
        const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
        INFO("idx=" << idx << " analytical=" << an << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
        CHECK(std::abs(an - fd) <= tol + 1e-6);
    }
}

TEST_CASE("DM force walker reports false for unowned theta",
          "[dphi_dtheta_walk][dm][force][routing][mblip]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    const std::size_t natoms = db.structures[0].natoms();

    std::vector<double> dphi(3 * natoms * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    /* CGRID2B is the D2 key — DM_mBlip doesn't own it. */
    const bool any = tadah::mlip::assemble_dphi_train_force_rows_dm(
        db, dm, "CGRID2B", /*arg_pos=*/1,
        dphi.data(), n_cols, force_row_start);
    CHECK_FALSE(any);
    for (double v : dphi) CHECK(v == 0.0);
}

TEST_CASE("DM walker reports false for unowned theta",
          "[dphi_dtheta_walk][dm][routing]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.7, 1.5};
    const std::vector<double> sgrid = {2.0, 3.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_ead_context_mlip(cgrid, sgrid, rcut, Lmax);
    DM_EAD dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    /* CGRID2B is the D2 key; DM_EAD doesn't own it. */
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "CGRID2B", /*arg_pos=*/1, dphi.data(), n_cols);
    CHECK_FALSE(any);
    for (double v : dphi) CHECK(v == 0.0);
}

// ============================================================================
// Kernel-arm walker — non-linear kernel features.
//
// Tests `assemble_dphi_train_energy_rows_d2_kern` against FD on a small
// 3-atom H cluster with D2_BP descriptor and Kern_RBF kernel with a
// hand-set basis (no EKM — basis is deterministic, not learned). The
// walker is exercised on:
//   (1) MPARAMS sigma (kernel hyperparameter; uses calc_dKdtheta).
//   (2) CGRID2B[1]    (descriptor hyperparameter; chains dK/daed with
//                      the existing D2 walker's dD/dtheta).
//   (3) "WATOM"       (unowned theta; returns false, dphi untouched).
//
// FD reference helper rebuilds Phi[s, b] = sum_atoms K(basis_b, aed_atom)
// directly via fresh D2_BP / Kern_RBF instances at perturbed theta, so
// the chain rule wiring inside the walker is fully exercised.
// ============================================================================

// Construct a deterministic 2-column basis matrix of `aed_size` rows.
// Each column is a smooth sin/cos vector — easy to read off, doesn't
// degenerate, and has no symmetry with D2_BP's built-in AED structure.
inline Matrix make_kern_basis(std::size_t aed_size, std::size_t nbasis) {
    Matrix b(aed_size, nbasis);
    for (std::size_t bb = 0; bb < nbasis; ++bb) {
        for (std::size_t k = 0; k < aed_size; ++k) {
            b(k, bb) = 0.3 * std::sin(0.7 * static_cast<double>(k + 1)
                                       + 1.1 * static_cast<double>(bb + 1));
        }
    }
    return b;
}

// FD reference: build the kernel-features row of Phi for the supplied
// D2 + kernel instances. Phi[s, b] = sum_atoms K(basis_b, aed_atom).
// Mirrors DM_Kern_Base::calc_phi_energy_row for an unprojected kernel
// matrix (no EKM applied).
inline std::vector<double>
build_phi_row_via_kernel(const D2_BP& d2,
                         const Kern_RBF& kern,
                         const Structure& st)
{
    const Matrix& basis = kern.get_basis();
    const std::size_t nbasis = basis.cols();
    const double rcut = d2.get_rcut();
    const double rcut_sq = rcut * rcut;

    std::vector<double> row(nbasis, 0.0);
    aed_type aed(d2.size());

    for (std::size_t i = 0; i < st.natoms(); ++i) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed.size()); ++k)
            aed[k] = 0.0;
        const Atom& a1 = st(i);
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            const double dx = a1.position[0] - a2pos[0];
            const double dy = a1.position[1] - a2pos[1];
            const double dz = a1.position[2] - a2pos[2];
            const double rij_sq = dx*dx + dy*dy + dz*dz;
            if (rij_sq >= rcut_sq) continue;
            const std::size_t neighIdx = st.nn->idx[i][jj];
            const int Zi = a1.Z;
            const int Zj = st(neighIdx).Z;
            const double rij = std::sqrt(rij_sq);
            d2.calc_aed(Zi, Zj, rij, rij_sq, aed, /*scale=*/0.5);
        }
        for (std::size_t b = 0; b < nbasis; ++b) {
            aed_type basis_col = basis.col(b);
            row[b] += kern(basis_col, aed);
        }
    }
    return row;
}

// Construct a Kern_RBF on a fresh Context with the supplied sigma, then
// attach a hand-set basis matrix. Mirrors the constructor path Kern_RBF
// expects (MPARAMS = sigma).
inline Kern_RBF make_rbf_with_basis(double sigma, const Matrix& basis) {
    Context kc;
    kc.add("MPARAMS", sigma);
    kc.add("VERBOSE", 0);
    Kern_RBF k(kc);
    k.set_basis(basis);
    return k;
}

TEST_CASE("kernel walker (D2_BP + Kern_RBF) matches FD on MPARAMS sigma",
          "[dphi_dtheta_walk][kern][parity][mparams]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;
    const double sigma = 1.2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t aed_size = d.size();
    REQUIRE(aed_size >= 1);
    Matrix basis = make_kern_basis(aed_size, /*nbasis=*/2);
    Kern_RBF kern = make_rbf_with_basis(sigma, basis);

    const std::size_t n_cols = basis.cols();
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_d2_kern(
        db, d, kern, "MPARAMS", /*arg_pos=*/1, dphi.data(), n_cols);
    REQUIRE(any);

    // FD: build the kernel-features Phi row at sigma ± h via a fresh
    // Kern_RBF instance at the perturbed sigma. D2 is unchanged.
    const double h = 1e-6;
    Kern_RBF kern_p = make_rbf_with_basis(sigma + h, basis);
    Kern_RBF kern_m = make_rbf_with_basis(sigma - h, basis);
    std::vector<double> row_p = build_phi_row_via_kernel(d, kern_p, db.structures[0]);
    std::vector<double> row_m = build_phi_row_via_kernel(d, kern_m, db.structures[0]);

    for (std::size_t b = 0; b < n_cols; ++b) {
        const double a  = dphi[b];
        const double fd = (row_p[b] - row_m[b]) / (2.0 * h);
        INFO("b=" << b << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-6 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-7);
    }
}

TEST_CASE("kernel walker (D2_BP + Kern_RBF) matches FD on CGRID2B[1]",
          "[dphi_dtheta_walk][kern][parity][cgrid]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;
    const double sigma = 1.2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    Matrix basis = make_kern_basis(d.size(), /*nbasis=*/2);
    Kern_RBF kern = make_rbf_with_basis(sigma, basis);

    const std::size_t n_cols = basis.cols();
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_d2_kern(
        db, d, kern, "CGRID2B", /*arg_pos=*/1, dphi.data(), n_cols);
    REQUIRE(any);

    // FD: perturb cgrid[0], rebuild D2_BP, and recompute the kernel Phi
    // row through the chain rule. Kern is unchanged (sigma fixed).
    const double h = 1e-5;
    auto cgrid_p = cgrid;  cgrid_p[0] += h;
    auto cgrid_m = cgrid;  cgrid_m[0] -= h;
    Context cp = make_bp_context(cgrid_p, sgrid, rcut);
    Context cm = make_bp_context(cgrid_m, sgrid, rcut);
    D2_BP d_p(cp);  Cut_Cos fc_p(rcut);  d_p.set_fcut(&fc_p, false);
    D2_BP d_m(cm);  Cut_Cos fc_m(rcut);  d_m.set_fcut(&fc_m, false);
    std::vector<double> row_p = build_phi_row_via_kernel(d_p, kern, db.structures[0]);
    std::vector<double> row_m = build_phi_row_via_kernel(d_m, kern, db.structures[0]);

    for (std::size_t b = 0; b < n_cols; ++b) {
        const double a  = dphi[b];
        const double fd = (row_p[b] - row_m[b]) / (2.0 * h);
        INFO("b=" << b << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-7);
    }
}

TEST_CASE("kernel walker reports false for unowned theta",
          "[dphi_dtheta_walk][kern][routing]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5};
    const std::vector<double> sgrid = {3.0};
    const double rcut = 2.5;
    const double sigma = 1.2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    Matrix basis = make_kern_basis(d.size(), /*nbasis=*/2);
    Kern_RBF kern = make_rbf_with_basis(sigma, basis);

    const std::size_t n_cols = basis.cols();
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    // "WATOM" with arg_pos=-1 is not owned by D2_BP::calc_dDdtheta (the
    // WATOM branch requires arg_pos >= 1 to resolve a species Z target)
    // nor by Kern_RBF::calc_dKdtheta — walker returns false, leaves
    // dphi untouched.
    const bool any = tadah::mlip::assemble_dphi_train_energy_rows_d2_kern(
        db, d, kern, "WATOM", /*arg_pos=*/-1, dphi.data(), n_cols);
    CHECK_FALSE(any);
    for (double v : dphi) CHECK(v == 0.0);
}

// ============================================================================
// WATOM walker tests. WATOM[1] is owned by every D2/DM
// descriptor that uses weights[Zj]; the walker should report true and
// the IFT chain should match FD.
// ============================================================================

namespace {

Context make_bp_context_with_watom(const std::vector<double>& cgrid,
                                   const std::vector<double>& sgrid,
                                   double rcut, double watom)
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", watom);
    c.add("INIT2B", "true");
    c.add("VERBOSE", 0);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) { if (i) ss << " "; ss << sgrid[i]; }
        c.add("SGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) { if (i) ss << " "; ss << cgrid[i]; }
        c.add("CGRID2B", ss.str());
    }
    c.add("RCUT2B", rcut);
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", static_cast<int>(cgrid.size()));
    c.add("TYPE2B", static_cast<int>(sgrid.size()));
    c.add("TYPE2B", "H");
    c.add("TYPE2B", "H");
    return c;
}

aed_type build_phi_row_via_calc_aed_with_watom(
        const std::vector<double>& cgrid,
        const std::vector<double>& sgrid,
        double rcut, double watom,
        const Structure& st)
{
    Context c = make_bp_context_with_watom(cgrid, sgrid, rcut, watom);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    aed_type row(d.size());
    for (std::size_t k = 0; k < static_cast<std::size_t>(row.size()); ++k) row[k] = 0.0;
    aed_type aed(d.size());
    const double rcut_sq = rcut * rcut;
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed.size()); ++k) aed[k] = 0.0;
        const Atom& a1 = st(i);
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            const double dx = a1.position[0] - a2pos[0];
            const double dy = a1.position[1] - a2pos[1];
            const double dz = a1.position[2] - a2pos[2];
            const double rij_sq = dx*dx + dy*dy + dz*dz;
            if (rij_sq >= rcut_sq) continue;
            const std::size_t neighIdx = st.nn->idx[i][jj];
            const int Zi = a1.Z;
            const int Zj = st(neighIdx).Z;
            const double rij = std::sqrt(rij_sq);
            d.calc_aed(Zi, Zj, rij, rij_sq, aed, /*scale=*/0.5);
        }
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed.size()); ++k)
            row[k] += aed[k];
    }
    return row;
}

std::vector<double>
build_phi_force_block_via_calc_all_with_watom(
        const std::vector<double>& cgrid,
        const std::vector<double>& sgrid,
        double rcut, double watom,
        const Structure& st)
{
    Context c = make_bp_context_with_watom(cgrid, sgrid, rcut, watom);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);
    const std::size_t n_cols = d.size();
    const std::size_t n_rows = 3 * st.natoms();
    std::vector<double> block(n_rows * n_cols, 0.0);

    std::vector<std::vector<tadah::core::fd_type>> fd_store(st.natoms());
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        fd_store[a].reserve(st.nn_size(a));
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj)
            fd_store[a].emplace_back(n_cols);
    }
    const double rcut_sq = rcut * rcut;
    aed_type aed_scratch(d.size());
    for (std::size_t i = 0; i < st.natoms(); ++i) {
        for (std::size_t k = 0; k < static_cast<std::size_t>(aed_scratch.size()); ++k)
            aed_scratch[k] = 0.0;
        const Atom& a1 = st(i);
        const int Zi = a1.Z;
        for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
            const Vec3d& a2pos = st.nn_pos(i, jj);
            const double dx = a1.position[0] - a2pos[0];
            const double dy = a1.position[1] - a2pos[1];
            const double dz = a1.position[2] - a2pos[2];
            const double rij_sq = dx*dx + dy*dy + dz*dz;
            if (rij_sq >= rcut_sq) continue;
            const std::size_t neighIdx = st.nn->idx[i][jj];
            const int Zj = st(neighIdx).Z;
            const double rij = std::sqrt(rij_sq);
            const double rij_inv = 1.0 / rij;
            tadah::core::fd_type& fij = fd_store[i][jj];
            d.calc_all(Zi, Zj, rij, rij_sq, aed_scratch, fij, /*scale=*/0.5);
            for (std::size_t n = 0; n < n_cols; ++n) {
                fij(n, 0) *= rij_inv;
                fij(n, 1) = fij(n, 0) * dy;
                fij(n, 2) = fij(n, 0) * dz;
                fij(n, 0) *= dx;
            }
        }
    }
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        for (std::size_t jj = 0; jj < st.nn_size(a); ++jj) {
            const std::size_t j   = st.nn->idx[a][jj];
            const std::size_t aa  = st.get_nn_iindex(a, j, jj);
            const tadah::core::fd_type& fij = fd_store[a][jj];
            const tadah::core::fd_type& fji = fd_store[j][aa];
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t n = 0; n < n_cols; ++n)
                    block[(3 * a + k) * n_cols + n] -= (fij(n, k) - fji(n, k));
        }
    }
    return block;
}

Context make_mblip_mb_context_mlip_with_watom(
        double cgrid_val, double sgrid_val,
        const std::vector<double>& cembfunc,
        const std::vector<double>& sembfunc,
        double rcut, double watom, int Lmax = 2)
{
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", watom);
    c.add("INITMB", "true");
    c.add("VERBOSE", 0);
    c.add("SGRIDMB", sgrid_val);
    c.add("CGRIDMB", cgrid_val);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cembfunc.size(); ++i) { if (i) ss << " "; ss << cembfunc[i]; }
        c.add("CEMBFUNC", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sembfunc.size(); ++i) { if (i) ss << " "; ss << sembfunc[i]; }
        c.add("SEMBFUNC", ss.str());
    }
    c.add("RCUTMB", rcut);
    c.add("TYPEMB", "DM_mBlip");
    c.add("TYPEMB", Lmax);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", static_cast<int>(cembfunc.size()));
    c.add("TYPEMB", static_cast<int>(sembfunc.size()));
    c.add("TYPEMB", "H"); c.add("TYPEMB", "H");
    return c;
}

}  // namespace

TEST_CASE("dPhi/dtheta walker matches FD on WATOM[1] (D2_BP, 3-atom H)",
          "[dphi_dtheta_walk][parity][watom]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context_with_watom(cgrid, sgrid, rcut, 1.0);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    REQUIRE(tadah::mlip::assemble_dphi_train_energy_rows_d2(
        db, d, "WATOM", /*arg_pos=*/1, dphi.data(), n_cols));

    const double h = 1e-5;
    aed_type row_p = build_phi_row_via_calc_aed_with_watom(cgrid, sgrid, rcut, 1.0 + h, db.structures[0]);
    aed_type row_m = build_phi_row_via_calc_aed_with_watom(cgrid, sgrid, rcut, 1.0 - h, db.structures[0]);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-6 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-9);
    }
}

TEST_CASE("dPhi/dtheta force walker matches FD on WATOM[1] (D2_BP)",
          "[dphi_dtheta_walk][force][parity][watom]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context_with_watom(cgrid, sgrid, rcut, 1.0);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();
    const std::size_t n_force_rows = 3 * db.structures[0].natoms();
    std::vector<double> dphi(n_force_rows * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    REQUIRE(tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "WATOM", /*arg_pos=*/1, dphi.data(), n_cols, force_row_start));

    const double h = 1e-5;
    auto block_p = build_phi_force_block_via_calc_all_with_watom(
        cgrid, sgrid, rcut, 1.0 + h, db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_all_with_watom(
        cgrid, sgrid, rcut, 1.0 - h, db.structures[0]);

    for (std::size_t idx = 0; idx < dphi.size(); ++idx) {
        const double an = dphi[idx];
        const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
        INFO("idx=" << idx << " analytical=" << an << " FD=" << fd);
        const double tol = 1e-6 * std::max({std::abs(an), std::abs(fd), 1e-9});
        CHECK(std::abs(an - fd) <= tol + 1e-9);
    }
}

TEST_CASE("DM walker (DM_mBlip) matches FD on WATOM[1]",
          "[dphi_dtheta_walk][dm][parity][watom][mblip]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5, sgrid_val = 3.0, rcut = 3.5;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const int Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip_with_watom(
        cgrid_val, sgrid_val, cembfunc, sembfunc, rcut, 1.0, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    std::vector<double> dphi(db.size() * n_cols, 0.0);
    REQUIRE(tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "WATOM", /*arg_pos=*/1, dphi.data(), n_cols));

    // FD: rebuild dm at WATOM ± h, re-walk the energy pipeline. Use
    // build_phi_row_via_dm_mblip helper inline.
    auto build_dm_row = [&](double watom_val) -> std::vector<double> {
        Context cw = make_mblip_mb_context_mlip_with_watom(
            cgrid_val, sgrid_val, cembfunc, sembfunc, rcut, watom_val, Lmax);
        DM_mEAD<F_Blip> dmw(cw);
        Cut_Cos fcw(rcut);
        dmw.set_fcut(&fcw, false);
        std::vector<double> row(n_cols, 0.0);
        const Structure& st = db.structures[0];
        const double rcut_sq = rcut * rcut;
        const std::size_t rho_total = dmw.rhoi_size() + dmw.rhoip_size();
        Vec3d delij;
        for (std::size_t i = 0; i < st.natoms(); ++i) {
            tadah::core::rho_type rhoi(static_cast<long>(rho_total));
            rhoi.set_zero();
            const Atom& a1 = st(i);
            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const Vec3d& a2pos = st.nn_pos(i, jj);
                delij[0] = a1.position[0] - a2pos[0];
                delij[1] = a1.position[1] - a2pos[1];
                delij[2] = a1.position[2] - a2pos[2];
                const double rij_sq = delij*delij;
                if (rij_sq >= rcut_sq) continue;
                const double rij = std::sqrt(rij_sq);
                dmw.calc_rho(1, 1, rij, rij_sq, delij, rhoi, 1.0);
            }
            aed_type aed(n_cols); aed.set_zero();
            dmw.calc_aed(rhoi, aed);
            for (std::size_t k = 0; k < n_cols; ++k) row[k] += aed[k];
        }
        return row;
    };

    const double h = 1e-5;
    auto row_p = build_dm_row(1.0 + h);
    auto row_m = build_dm_row(1.0 - h);

    for (std::size_t k = 0; k < n_cols; ++k) {
        const double a  = dphi[k];
        const double fd = (row_p[k] - row_m[k]) / (2.0 * h);
        INFO("k=" << k << " analytical=" << a << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(a), std::abs(fd), 1e-9});
        CHECK(std::abs(a - fd) <= tol + 1e-6);
    }
}

TEST_CASE("DM force walker (DM_mBlip) matches FD on WATOM[1]",
          "[dphi_dtheta_walk][dm][force][parity][watom][mblip]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5, sgrid_val = 3.0, rcut = 3.5;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const int Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip_with_watom(
        cgrid_val, sgrid_val, cembfunc, sembfunc, rcut, 1.0, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();
    const std::size_t natoms = db.structures[0].natoms();
    std::vector<double> dphi(3 * natoms * n_cols, 0.0);
    std::vector<std::size_t> force_row_start = {0};
    REQUIRE(tadah::mlip::assemble_dphi_train_force_rows_dm(
        db, dm, "WATOM", /*arg_pos=*/1, dphi.data(), n_cols, force_row_start));

    const double h = 1e-5;
    Context cp = make_mblip_mb_context_mlip_with_watom(
        cgrid_val, sgrid_val, cembfunc, sembfunc, rcut, 1.0 + h, Lmax);
    Context cm = make_mblip_mb_context_mlip_with_watom(
        cgrid_val, sgrid_val, cembfunc, sembfunc, rcut, 1.0 - h, Lmax);
    auto block_p = build_phi_force_block_via_calc_dXijdri_dm(cp, rcut, db.structures[0]);
    auto block_m = build_phi_force_block_via_calc_dXijdri_dm(cm, rcut, db.structures[0]);

    for (std::size_t idx = 0; idx < dphi.size(); ++idx) {
        const double an = dphi[idx];
        const double fd = (block_p[idx] - block_m[idx]) / (2.0 * h);
        INFO("idx=" << idx << " analytical=" << an << " FD=" << fd);
        const double tol = 1e-5 * std::max({std::abs(an), std::abs(fd), 1e-9});
        CHECK(std::abs(an - fd) <= tol + 1e-6);
    }
}

// ============================================================================
// Layer-1.5 parity tests for the combined (energy + force)
// walker. The combined walker fuses the per-atom / per-pair iteration of the
// legacy two-pass walker into one structure traversal sharing scratch state
// (and, for DM, sharing the per-atom rho_i / drho_i / F'(rho_i) buffers).
// These tests assert that the combined walker produces byte-identical
// `dphi_energy_out` and `dphi_force_out` to the two legacy walkers run back
// to back on the same inputs — the load-bearing correctness gate for wiring
// it into compute_analytical_gradient_from_trainer's per-θ loop.
// ============================================================================

TEST_CASE("combined walker D2_BP CGRID2B[1] matches legacy two-pass byte-identically",
          "[dphi_dtheta_walk][parity][combined][cgrid][force][d2]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols       = d.size();
    const std::size_t natoms       = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;
    const std::vector<std::size_t> force_row_start = {0};

    /* Legacy two-pass: energy then force, separate buffers. */
    std::vector<double> legacy_energy(db.size() * n_cols, 0.0);
    std::vector<double> legacy_force (n_force_rows * n_cols, 0.0);
    const bool legacy_ok_e = tadah::mlip::assemble_dphi_train_energy_rows_d2(
        db, d, "CGRID2B", /*arg_pos=*/1,
        legacy_energy.data(), n_cols);
    const bool legacy_ok_f = tadah::mlip::assemble_dphi_train_force_rows_d2(
        db, d, "CGRID2B", /*arg_pos=*/1,
        legacy_force.data(),  n_cols, force_row_start);
    REQUIRE(legacy_ok_e);
    REQUIRE(legacy_ok_f);

    /* Combined: one walker writing to both buffers. */
    std::vector<double> both_energy(db.size() * n_cols, 0.0);
    std::vector<double> both_force (n_force_rows * n_cols, 0.0);
    const auto pr = tadah::mlip::assemble_dphi_train_both_rows_d2(
        db, d, "CGRID2B", /*arg_pos=*/1,
        both_energy.data(), both_force.data(),
        n_cols, force_row_start);
    CHECK(pr.first  == legacy_ok_e);
    CHECK(pr.second == legacy_ok_f);

    REQUIRE(legacy_energy.size() == both_energy.size());
    REQUIRE(legacy_force.size()  == both_force.size());
    for (std::size_t i = 0; i < legacy_energy.size(); ++i) {
        INFO("energy idx=" << i
             << " legacy=" << legacy_energy[i]
             << " combined=" << both_energy[i]);
        REQUIRE(legacy_energy[i] == both_energy[i]);
    }
    for (std::size_t i = 0; i < legacy_force.size(); ++i) {
        INFO("force idx=" << i
             << " legacy=" << legacy_force[i]
             << " combined=" << both_force[i]);
        REQUIRE(legacy_force[i] == both_force[i]);
    }
}

TEST_CASE("combined walker D2_BP CGRID2B[1] energy-only (force_out=nullptr) "
          "matches legacy energy walker byte-identically",
          "[dphi_dtheta_walk][parity][combined][cgrid][d2]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_three_atom_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_bp_context(cgrid, sgrid, rcut);
    D2_BP d(c);
    Cut_Cos fc(rcut);
    d.set_fcut(&fc, false);

    const std::size_t n_cols = d.size();

    std::vector<double> legacy_energy(db.size() * n_cols, 0.0);
    const bool legacy_ok_e = tadah::mlip::assemble_dphi_train_energy_rows_d2(
        db, d, "CGRID2B", 1, legacy_energy.data(), n_cols);
    REQUIRE(legacy_ok_e);

    std::vector<double> both_energy(db.size() * n_cols, 0.0);
    const auto pr = tadah::mlip::assemble_dphi_train_both_rows_d2(
        db, d, "CGRID2B", 1, both_energy.data(), /*force_out=*/nullptr,
        n_cols, /*force_row_start=*/{});
    CHECK(pr.first  == legacy_ok_e);
    CHECK(pr.second == false);

    for (std::size_t i = 0; i < legacy_energy.size(); ++i) {
        INFO("idx=" << i << " legacy=" << legacy_energy[i]
                    << " combined=" << both_energy[i]);
        REQUIRE(legacy_energy[i] == both_energy[i]);
    }
}

TEST_CASE("combined walker DM_mBlip CGRIDMB[1] matches legacy two-pass byte-identically",
          "[dphi_dtheta_walk][parity][combined][dm][mblip][force]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols       = dm.size();
    const std::size_t natoms       = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;
    const std::vector<std::size_t> force_row_start = {0};

    std::vector<double> legacy_energy(db.size() * n_cols, 0.0);
    std::vector<double> legacy_force (n_force_rows * n_cols, 0.0);
    const bool legacy_ok_e = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1, legacy_energy.data(), n_cols);
    const bool legacy_ok_f = tadah::mlip::assemble_dphi_train_force_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1, legacy_force.data(),  n_cols,
        force_row_start);
    REQUIRE(legacy_ok_e);
    REQUIRE(legacy_ok_f);

    std::vector<double> both_energy(db.size() * n_cols, 0.0);
    std::vector<double> both_force (n_force_rows * n_cols, 0.0);
    const auto pr = tadah::mlip::assemble_dphi_train_both_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1,
        both_energy.data(), both_force.data(),
        n_cols, force_row_start);
    CHECK(pr.first  == legacy_ok_e);
    CHECK(pr.second == legacy_ok_f);

    REQUIRE(legacy_energy.size() == both_energy.size());
    REQUIRE(legacy_force.size()  == both_force.size());
    for (std::size_t i = 0; i < legacy_energy.size(); ++i) {
        INFO("energy idx=" << i
             << " legacy=" << legacy_energy[i]
             << " combined=" << both_energy[i]);
        REQUIRE(legacy_energy[i] == both_energy[i]);
    }
    for (std::size_t i = 0; i < legacy_force.size(); ++i) {
        INFO("force idx=" << i
             << " legacy=" << legacy_force[i]
             << " combined=" << both_force[i]);
        REQUIRE(legacy_force[i] == both_force[i]);
    }
}

TEST_CASE("combined walker DM_mBlip CGRIDMB[1] energy-only (force_out=nullptr) "
          "matches legacy energy walker byte-identically",
          "[dphi_dtheta_walk][parity][combined][dm][mblip]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols = dm.size();

    std::vector<double> legacy_energy(db.size() * n_cols, 0.0);
    const bool legacy_ok_e = tadah::mlip::assemble_dphi_train_energy_rows_dm(
        db, dm, "CGRIDMB", 1, legacy_energy.data(), n_cols);
    REQUIRE(legacy_ok_e);

    std::vector<double> both_energy(db.size() * n_cols, 0.0);
    const auto pr = tadah::mlip::assemble_dphi_train_both_rows_dm(
        db, dm, "CGRIDMB", 1, both_energy.data(), /*force_out=*/nullptr,
        n_cols, /*force_row_start=*/{});
    CHECK(pr.first  == legacy_ok_e);
    CHECK(pr.second == false);

    for (std::size_t i = 0; i < legacy_energy.size(); ++i) {
        INFO("idx=" << i << " legacy=" << legacy_energy[i]
                    << " combined=" << both_energy[i]);
        REQUIRE(legacy_energy[i] == both_energy[i]);
    }
}

// ============================================================================
// Layer-1.5 parity tests for the cached DM walker. The cached
// walker reads rho_i / F'(rho_i) from a pre-built DMRhoCache instead of
// recomputing them; the math is otherwise identical to the non-cached
// combined walker, so the outputs must be byte-identical.
// ============================================================================

TEST_CASE("cached DM walker (DM_mBlip CGRIDMB[1]) matches non-cached combined "
          "byte-identically",
          "[dphi_dtheta_walk][parity][cached][dm][mblip][force]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols       = dm.size();
    const std::size_t natoms       = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;
    const std::vector<std::size_t> force_row_start = {0};

    /* Non-cached combined walker as reference. */
    std::vector<double> ref_energy(db.size() * n_cols, 0.0);
    std::vector<double> ref_force (n_force_rows * n_cols, 0.0);
    const auto pr_ref = tadah::mlip::assemble_dphi_train_both_rows_dm(
        db, dm, "CGRIDMB", /*arg_pos=*/1,
        ref_energy.data(), ref_force.data(),
        n_cols, force_row_start);
    REQUIRE(pr_ref.first);
    REQUIRE(pr_ref.second);

    /* Build rho cache once, then run the cached walker. */
    tadah::mlip::DMRhoCache cache;
    tadah::mlip::build_dm_rho_cache(db, dm, cache, n_cols);
    REQUIRE(cache.rhoi_per_struct.size() == db.size());
    REQUIRE(cache.rhoi_per_struct[0].size() == natoms);

    std::vector<double> cached_energy(db.size() * n_cols, 0.0);
    std::vector<double> cached_force (n_force_rows * n_cols, 0.0);
    const auto pr_cached = tadah::mlip::assemble_dphi_train_both_rows_dm_cached(
        db, dm, cache, "CGRIDMB", /*arg_pos=*/1,
        cached_energy.data(), cached_force.data(),
        n_cols, force_row_start);
    CHECK(pr_cached.first  == pr_ref.first);
    CHECK(pr_cached.second == pr_ref.second);

    for (std::size_t i = 0; i < ref_energy.size(); ++i) {
        INFO("energy idx=" << i
             << " ref=" << ref_energy[i]
             << " cached=" << cached_energy[i]);
        REQUIRE(ref_energy[i] == cached_energy[i]);
    }
    for (std::size_t i = 0; i < ref_force.size(); ++i) {
        INFO("force idx=" << i
             << " ref=" << ref_force[i]
             << " cached=" << cached_force[i]);
        REQUIRE(ref_force[i] == cached_force[i]);
    }
}

TEST_CASE("cached DM walker (DM_mBlip CEMBFUNC[1]) matches non-cached combined "
          "byte-identically",
          "[dphi_dtheta_walk][parity][cached][dm][mblip][force][embedding]")
{
    PeriodicTable::initialize();

    const double cgrid_val = 0.5;
    const double sgrid_val = 3.0;
    const std::vector<double> cembfunc = {0.05, 0.15};
    const std::vector<double> sembfunc = {20.0, 20.0};
    const double rcut = 3.5;
    const int    Lmax = 2;

    StructureDB db;
    db.dbidx = {0, 1};
    db.structures.push_back(make_four_atom_off_axis_structure());
    NNFinder finder;
    finder.calc(db.structures[0], rcut);

    Context c = make_mblip_mb_context_mlip(cgrid_val, sgrid_val,
                                            cembfunc, sembfunc, rcut, Lmax);
    DM_mEAD<F_Blip> dm(c);
    Cut_Cos fc(rcut);
    dm.set_fcut(&fc, false);

    const std::size_t n_cols       = dm.size();
    const std::size_t natoms       = db.structures[0].natoms();
    const std::size_t n_force_rows = 3 * natoms;
    const std::vector<std::size_t> force_row_start = {0};

    std::vector<double> ref_energy(db.size() * n_cols, 0.0);
    std::vector<double> ref_force (n_force_rows * n_cols, 0.0);
    const auto pr_ref = tadah::mlip::assemble_dphi_train_both_rows_dm(
        db, dm, "CEMBFUNC", /*arg_pos=*/1,
        ref_energy.data(), ref_force.data(),
        n_cols, force_row_start);
    REQUIRE(pr_ref.first);
    REQUIRE(pr_ref.second);

    tadah::mlip::DMRhoCache cache;
    tadah::mlip::build_dm_rho_cache(db, dm, cache, n_cols);

    std::vector<double> cached_energy(db.size() * n_cols, 0.0);
    std::vector<double> cached_force (n_force_rows * n_cols, 0.0);
    const auto pr_cached = tadah::mlip::assemble_dphi_train_both_rows_dm_cached(
        db, dm, cache, "CEMBFUNC", /*arg_pos=*/1,
        cached_energy.data(), cached_force.data(),
        n_cols, force_row_start);
    CHECK(pr_cached.first  == pr_ref.first);
    CHECK(pr_cached.second == pr_ref.second);

    for (std::size_t i = 0; i < ref_energy.size(); ++i) {
        INFO("energy idx=" << i
             << " ref=" << ref_energy[i]
             << " cached=" << cached_energy[i]);
        REQUIRE(ref_energy[i] == cached_energy[i]);
    }
    for (std::size_t i = 0; i < ref_force.size(); ++i) {
        INFO("force idx=" << i
             << " ref=" << ref_force[i]
             << " cached=" << cached_force[i]);
        REQUIRE(ref_force[i] == cached_force[i]);
    }
}

// ============================================================================
// Combined walker of a DC_Selector with a two-body and a many-body
// descriptor: the claims of the two joined.
// ============================================================================

namespace {

/* A two-atom bcc Ta cell (a = 3.3 A), the second atom moved off its site,
 * with its neighbour list to 5 A: pairs at many distances inside the
 * cutoffs of 4.5 A below. */
StructureDB two_atom_ta_cell()
{
    Structure st;
    st.cell.load(3.3, 0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 3.3);
    st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("Ta"), 1.72, 1.61, 1.66, 0.0, 0.0, 0.0));
    NNFinder().calc(st, 5.0);
    StructureDB db;
    db.add(st);
    return db;
}

/* The claims (energy rows, force rows) of the descriptors of the
 * training configuration with TYPE2B D2_BP and the many-body descriptor
 * `typemb` (with its `lines`, and RCTYPEMB Cut_Cos, RCUTMB 4.5; an empty
 * `typemb`: `lines` are the whole many-body term) for `key` (index 1) on
 * two_atom_ta_cell(), the force rows asked when `force`: by the combined
 * walker, and by the combined walker with the rho cache. */
struct JoinedClaims {
    std::pair<bool, bool> plain, cached;
};
JoinedClaims joined_claims(const std::string& typemb,
                           const std::string& lines,
                           const std::string& key, bool force)
{
    std::istringstream in(
        "ATOM Ta\nWATOM 1.0\nMODEL M_BLR BF_Linear\nVERBOSE 0\n"
        "INIT2B true\nTYPE2B D2_BP 2 2 Ta Ta\nCGRID2B 1.0 2.0\n"
        "SGRID2B 1.0 1.2\nRCTYPE2B Cut_Cos\nRCUT2B 4.5\n"
        "INITMB true\n" +
        (typemb.empty() ? lines
                        : "TYPEMB " + typemb + "\n" + lines +
                              "RCTYPEMB Cut_Cos\nRCUTMB 4.5\n"));
    Context c = tadah::core::CoreUserConfigParser().parse(in);
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer trainer(c, ws);
    const StructureDB db = two_atom_ta_cell();
    const std::size_t n_cols = trainer.DCS.descriptor_size();
    std::vector<double> energy(n_cols, 0.0);
    std::vector<double> rows(3 * db(0).natoms() * n_cols, 0.0);
    double* force_rows = force ? rows.data() : nullptr;
    JoinedClaims out;
    out.plain = tadah::mlip::assemble_dphi_train_both_rows(
        db, trainer.DCS, key, 1, energy.data(), force_rows, n_cols, {0});
    tadah::mlip::DMRhoCache cache;
    tadah::mlip::build_dm_rho_cache(db, *trainer.DCS.dmb, cache, n_cols);
    energy.assign(n_cols, 0.0);
    rows.assign(rows.size(), 0.0);
    out.cached = tadah::mlip::assemble_dphi_train_both_rows_cached(
        db, trainer.DCS, cache, key, 1, energy.data(), force_rows, n_cols,
        {0});
    return out;
}

}  // namespace

/* Pins: the combined walker of a two-body and a many-body descriptor, with
 * and without the rho cache, claims the rows of a parameter only when every
 * descriptor that depends on it or claims its energy rows claims them too:
 *   - WATOM, which D2_BP and DM_Blip both read, D2_BP differentiating it in
 *     the energy and the force rows and DM_Blip in the energy rows only:
 *     (energy, force) = (true, false), the force rows asked or not;
 *   - CGRID2B, of D2_BP alone: (true, true); CGRIDMB, of DM_Blip alone:
 *     (true, false);
 *   - WATOM with DM_mBlip, which differentiates it in the force rows too:
 *     (true, true);
 *   - WATOM with DM_REAM, DM_PS_SQ, a DESCRIPTOR block (DM_Embed) or
 *     DM_mJoin (DM_Blip and DM_REAM joined), each of which reads WATOM and
 *     has no derivative with respect to it: (false, false); CGRID2B beside
 *     DM_REAM: (true, true).
 * Before the fix: D2_BP + DM_Blip claimed WATOM for the force rows,
 * (true, true): the claims of the two descriptors were joined by an OR, so
 * the force rows of the D2_BP part hid the missing DM_Blip part, and the
 * analytical gradient of tadah hpo took the result as the derivative of
 * the force rows. The same OR claimed WATOM beside DM_REAM, DM_PS_SQ, the
 * block and DM_mJoin, (true, true), rows without the part of the
 * many-body descriptor: those cases failed. */
TEST_CASE("combined walker: a parameter of a two-body and a many-body "
          "descriptor has its rows claimed only when each descriptor that "
          "depends on it or claims its energy rows claims them",
          "[dphi_dtheta_walk][combined][cached][watom][force]")
{
    PeriodicTable::initialize();
    using Claims = std::pair<bool, bool>;
    const std::string blip = "DM_Blip 0 2 2 Ta Ta";
    const std::string blip_grid = "CGRIDMB 2.0 3.0\nSGRIDMB 1.0 1.0\n";
    const std::string mblip = "DM_mBlip 0 1 1 2 2 Ta Ta";
    const std::string mblip_grid =
        "CGRIDMB 0.0\nSGRIDMB 0.2\nCEMBFUNC 1.0 2.0\nSEMBFUNC 1.0 1.0\n";
    const std::string ream = "DM_REAM 3 3 3 3 5 5 0 0 Ta Ta";
    const std::string ream_grid =
        "CGRIDMB 2.8 3.6 4.4\nSGRIDMB 3.0 3.0 3.0\nAMPGRIDMB 0.5 0.3 0.1\n"
        "CEMBFUNC -0.2 0.3 0.8 1.3 1.8\nSEMBFUNC 2.0 2.0 2.0 2.0 2.0\n"
        "RHOFORM SIGNED\n";
    const std::string embed =
        "DESCRIPTOR ta DM_Embed\nPAIRS Ta Ta\nDENSITY\nBASIS Blip5\n"
        "FORM SIGNED\nCENTRES 2.8 3.6 4.4\nETA 3 3 3\n"
        "AMPLITUDES 0.5 0.3 0.1\nCUTOFF\nTYPE Cut_Cos\nRADIUS 4.5\n"
        "ENDCUTOFF\nENDDENSITY\nEMBEDDING\nBASIS Gaussian\n"
        "CENTRES 0 0.5 1\nETA 2 2 2\nREFERENCE ZERO\nENDEMBEDDING\n"
        "ENDDESCRIPTOR\n";
    const std::string mjoin =
        "TYPEMB DM_mJoin\nTYPEMB DM_Blip 0 2 2 Ta Ta\nTYPEMB " + ream +
        "\nRCTYPEMB Cut_Cos Cut_Cos\nRCUTMB 4.5 4.5\n"
        "CGRIDMB 2.0 3.0 2.8 3.6 4.4\nSGRIDMB 1.0 1.0 3.0 3.0 3.0\n"
        "AMPGRIDMB 0.5 0.3 0.1\nCEMBFUNC -0.2 0.3 0.8 1.3 1.8\n"
        "SEMBFUNC 2.0 2.0 2.0 2.0 2.0\nRHOFORM SIGNED\n";
    struct Case {
        const char* name;
        std::string typemb, lines, key;
        bool force;
        Claims expected;
    };
    const std::vector<Case> cases = {
        {"D2_BP + DM_Blip, WATOM, force rows", blip, blip_grid, "WATOM",
         true, {true, false}},
        {"D2_BP + DM_Blip, WATOM, energy rows", blip, blip_grid, "WATOM",
         false, {true, false}},
        {"D2_BP + DM_Blip, CGRID2B, force rows", blip, blip_grid, "CGRID2B",
         true, {true, true}},
        {"D2_BP + DM_Blip, CGRIDMB, force rows", blip, blip_grid, "CGRIDMB",
         true, {true, false}},
        {"D2_BP + DM_mBlip, WATOM, force rows", mblip, mblip_grid, "WATOM",
         true, {true, true}},
        {"D2_BP + DM_REAM, WATOM, force rows", ream, ream_grid, "WATOM",
         true, {false, false}},
        {"D2_BP + DM_REAM, WATOM, energy rows", ream, ream_grid, "WATOM",
         false, {false, false}},
        {"D2_BP + DM_REAM, CGRID2B, force rows", ream, ream_grid, "CGRID2B",
         true, {true, true}},
        {"D2_BP + DM_PS_SQ, WATOM, force rows", "DM_PS_SQ 1 2 2 Ta Ta",
         blip_grid, "WATOM", true, {false, false}},
        {"D2_BP + a DESCRIPTOR block, WATOM, force rows", "", embed, "WATOM",
         true, {false, false}},
        {"D2_BP + DM_mJoin of DM_Blip and DM_REAM, WATOM, force rows", "",
         mjoin, "WATOM", true, {false, false}},
    };
    for (const Case& c : cases) {
        INFO(c.name);
        const JoinedClaims got =
            joined_claims(c.typemb, c.lines, c.key, c.force);
        CHECK(got.plain == c.expected);
        CHECK(got.cached == c.expected);
    }
}
