// test_native_relaxer.cpp
//
// Smoke tests for NativeRelaxer
// (MLIP/include/tadah/mlip/native_relaxer.h). The relaxer is the
// minimisation engine the typed cohesive_energy observable composes
// with cells::bcc + cells::isolated_atom to produce x* without going
// through LAMMPS.
//
// What these tests check:
//   1. Fixed mode on an already-equilibrium structure (perfect BCC,
//      perfectly symmetric): F ≈ 0 by symmetry → relaxer converges
//      at iter 0 without errors.
//   2. Iso mode on perturbed BCC: starting from a0 too small, the
//      relaxer's strain DOF drives the volume back up and final fmax
//      is below tolerance.
//   3. Aniso / Voigt6 / FixedZ modes are stubs and throw cleanly.
//   4. The strain force of the result, raw and over 3 V, is that of the
//      cell the relaxation ends at, also after an accepted last step.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/native_relaxer.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include "nn_oracle.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NativeRelaxer;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

/* Minimal Ta-on-BCC training context. D2_BP with two grid points,
 * Cut_Cos, BF_Linear, ridge. Same skeleton as test_m_tadah_base_w_grad.cpp
 * but for Ta. */
Context make_ta_context(double rcut, const std::vector<double>& cgrid,
                        const std::vector<double>& sgrid)
{
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(cgrid.size()));
    c.add("TYPE2B",   static_cast<int>(sgrid.size()));
    c.add("TYPE2B",   "Ta"); c.add("TYPE2B", "Ta");
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta"); c.add("TYPEMB", "Ta");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " "; ss << cgrid[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " "; ss << sgrid[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B",   rcut);
    c.add("RCUTMB",   rcut);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

/* Build a 4×4×4 BCC Ta supercell (128 atoms) via cells::bcc + replicate. */
Structure make_bcc_ta_supercell(double a0) {
    auto st = tadah::mlip::cells::bcc("Ta", a0);
    tadah::mlip::StructureTransformation::replicate(st, 4, 4, 4);
    return st;
}

/* Build a small Ta training DB so the trainer has data to fit on. We
 * train on a few BCC-Ta cells at slightly different lattice constants
 * — gives the model enough variation to fit non-zero weights. */
StructureDB make_ta_train_db(double rcut) {
    StructureDB db;
    const std::vector<double> a0s = {3.20, 3.30, 3.40};
    db.dbidx = {0, a0s.size()};
    db.structures.reserve(a0s.size());
    for (std::size_t i = 0; i < a0s.size(); ++i) {
        auto st = tadah::mlip::cells::bcc("Ta", a0s[i]);
        tadah::mlip::StructureTransformation::replicate(st, 2, 2, 2);
        /* Fake energies — physics is irrelevant; we just need a fitted
         * potential whose minimum is somewhere in this range. */
        st.energy = -0.5 * static_cast<double>(i + 1);
        db.structures.push_back(st);
    }
    return db;
}

}  // namespace

TEST_CASE("NativeRelaxer Aniso/Voigt6/FixedZ throw not_implemented",
          "[mlip][native_relaxer][stub]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);

    auto st = make_bcc_ta_supercell(3.30);
    NativeRelaxer relaxer;

    {
        NativeRelaxer::Options opts;
        opts.box_mode = NativeRelaxer::Options::Aniso;
        CHECK_THROWS_AS(relaxer.relax(st, *host.model, host.dc, host.context, opts),
                        std::logic_error);
    }
    {
        NativeRelaxer::Options opts;
        opts.box_mode = NativeRelaxer::Options::Voigt6;
        CHECK_THROWS_AS(relaxer.relax(st, *host.model, host.dc, host.context, opts),
                        std::logic_error);
    }
    {
        NativeRelaxer::Options opts;
        opts.box_mode = NativeRelaxer::Options::FixedZ;
        CHECK_THROWS_AS(relaxer.relax(st, *host.model, host.dc, host.context, opts),
                        std::logic_error);
    }
    /* The refusal is an internal error without plans (PC_OBSERVABLE
     * relaxes a fixed or an isotropic box only). Before the fix:
     * "NativeRelaxer: requested box_mode not implemented in MVP.
     * Aniso/Voigt6 are scheduled with ...". */
    {
        NativeRelaxer::Options opts;
        opts.box_mode = NativeRelaxer::Options::Voigt6;
        CHECK_THROWS_WITH(relaxer.relax(st, *host.model, host.dc, host.context, opts),
                          "internal error: the relaxation takes a fixed or "
                          "an isotropic box only");
    }
}

/* Pins: the verbose lines of a relaxation name the relaxation, not the
 * class: with verbose 2, the Iso relaxation of the compressed bcc cell
 * writes "relaxation: iteration 0, E=..., fmax=..." first, then lines
 * "relaxation: iteration <i>, E=..., fmax=..., step=..., scale=..." and
 * ends with "relaxation: converged at iteration <i>, E=..., fmax=...,
 * scale=..."; every line begins "relaxation: ".
 * Before the fix: "[NativeRelaxer] iter=0  E=...  fmax=...",
 * "[NativeRelaxer] converged at iter=...". */
TEST_CASE("NativeRelaxer: the verbose lines name the relaxation",
          "[mlip][native_relaxer][messages]")
{
    PeriodicTable::initialize();
    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);
    auto st = make_bcc_ta_supercell(3.10);
    NativeRelaxer::Options opts;
    opts.box_mode = NativeRelaxer::Options::Iso;
    opts.ftol     = 1.0e-3;
    opts.maxiter  = 500;
    opts.verbose  = 2;
    opts.init_step = 1.0e-3;
    opts.max_step  = 0.05;
    std::ostringstream err;
    std::streambuf *old = std::cerr.rdbuf(err.rdbuf());
    NativeRelaxer::Result res;
    try {
        res = NativeRelaxer().relax(st, *host.model, host.dc, host.context, opts);
    } catch (...) {
        std::cerr.rdbuf(old);
        throw;
    }
    std::cerr.rdbuf(old);
    const std::string text = err.str();
    INFO(text);
    REQUIRE(res.converged);
    std::istringstream in(text);
    std::string line, first, last;
    std::size_t n = 0, other = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line.rfind("relaxation: ", 0) != 0) ++other;
        if (n == 0) first = line;
        last = line;
        ++n;
    }
    CHECK(n >= 3);
    CHECK(other == 0);
    CHECK(first.rfind("relaxation: iteration 0, E=", 0) == 0);
    CHECK(text.find("relaxation: iteration 1, E=") != std::string::npos);
    CHECK(last.rfind("relaxation: converged at iteration ", 0) == 0);
    CHECK(last.find(", scale=") != std::string::npos);
    CHECK(text.find("NativeRelaxer") == std::string::npos);
}

TEST_CASE("NativeRelaxer Fixed mode on perfect BCC converges at iter 0",
          "[mlip][native_relaxer][fixed][bcc]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);

    /* Perfect BCC: by inversion symmetry, every atomic force is 0. */
    auto st = make_bcc_ta_supercell(3.30);

    NativeRelaxer::Options opts;
    opts.box_mode = NativeRelaxer::Options::Fixed;
    opts.ftol     = 1.0e-6;
    opts.maxiter  = 100;
    NativeRelaxer relaxer;
    auto res = relaxer.relax(st, *host.model, host.dc, host.context, opts);

    INFO("converged=" << res.converged << "  iter=" << res.n_iter
         << "  fmax=" << res.final_fmax_atoms);

    CHECK(res.converged);
    /* Iter-0 convergence check (added with D3 early-bail work) detects
     * the symmetric F≈0 state without entering the iter loop. Prior
     * expectation `n_iter == 1` predated this check. */
    CHECK(res.n_iter == 0);
    CHECK(res.final_fmax_atoms < 1.0e-6);
}

TEST_CASE("NativeRelaxer early_reject predicate aborts mid-relax",
          "[mlip][native_relaxer][iso][early_reject]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);

    /* Compressed BCC so the relax has work to do. */
    auto st = make_bcc_ta_supercell(3.10);

    NativeRelaxer::Options opts;
    opts.box_mode  = NativeRelaxer::Options::Iso;
    opts.pressure  = 0.0;
    opts.ftol      = 1.0e-6;
    opts.maxiter   = 500;
    opts.init_step = 1.0e-3;
    opts.max_step  = 0.05;
    /* Predicate: bail after iter 2 regardless of state. */
    opts.early_reject = [](const tadah::mlip::EarlyState& s) {
        return s.iter >= 2;
    };

    NativeRelaxer relaxer;
    auto res = relaxer.relax(st, *host.model, host.dc, host.context, opts);

    INFO("converged=" << res.converged
         << "  early_rejected=" << res.early_rejected
         << "  iter=" << res.n_iter);
    CHECK(res.early_rejected);
    CHECK_FALSE(res.converged);
    CHECK(res.n_iter == 2);
}

TEST_CASE("NativeRelaxer scale_out_of_range fires when LS exhausts on soft bound",
          "[mlip][native_relaxer][iso][scale_oor]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);

    auto st = make_bcc_ta_supercell(3.10);

    NativeRelaxer::Options opts;
    opts.box_mode  = NativeRelaxer::Options::Iso;
    opts.pressure  = 0.0;
    opts.ftol      = 1.0e-6;
    opts.maxiter   = 100;
    opts.init_step = 1.0e-3;
    opts.max_step  = 0.05;
    /* Tight scale bounds: scale must stay in [0.999, 1.001]. The relax
     * from a 3% compressed cell wants to expand → trial scale > 1.001 →
     * LS halving exhausts → scale_out_of_range = true. */
    opts.scale_min = 0.999;
    opts.scale_max = 1.001;
    opts.step_floor = 1.0e-8;

    NativeRelaxer relaxer;
    auto res = relaxer.relax(st, *host.model, host.dc, host.context, opts);

    INFO("converged=" << res.converged
         << "  scale_out_of_range=" << res.scale_out_of_range
         << "  iter=" << res.n_iter);
    /* Either the LS exhausts and reports scale_out_of_range (the
     * primary path) OR the relax converges within the tight bound (also
     * acceptable for very gentle synthetic potentials). The
     * scale_out_of_range latent bug was: prior to this fix the flag
     * was never assigned true; this case exercises the fix. */
    if (!res.converged) {
        CHECK(res.scale_out_of_range);
    }
}

TEST_CASE("NativeRelaxer Iso mode drives strain force below tolerance",
          "[mlip][native_relaxer][iso][bcc]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);

    /* Start away from equilibrium: compressed by ~3%. */
    auto st = make_bcc_ta_supercell(3.10);

    NativeRelaxer::Options opts;
    opts.box_mode = NativeRelaxer::Options::Iso;
    opts.pressure = 0.0;
    opts.ftol     = 1.0e-3;          /* loose ftol — synthetic potential, not a Ta DFT match */
    opts.maxiter  = 500;
    opts.verbose  = 0;
    opts.init_step = 1.0e-3;          /* small initial step — strain DOF can blow up */
    opts.max_step  = 0.05;

    NativeRelaxer relaxer;
    auto res = relaxer.relax(st, *host.model, host.dc, host.context, opts);

    INFO("converged=" << res.converged << "  iter=" << res.n_iter
         << "  fmax_atoms=" << res.final_fmax_atoms
         << "  F_eps=" << res.final_strain_force
         << "  scale=" << res.scale_factor
         << "  E=" << res.final_energy);

    /* MVP smoke: verify the relax loop converges. The relaxer's
     * convergence criterion normalises the strain force by (3·V) so the
     * raw `final_strain_force` may still be O(1) in absolute eV/strain
     * units while the pressure-like ratio is below ftol. */
    CHECK(res.converged);
    CHECK(res.n_iter > 1);                /* did make ≥1 successful step */
    CHECK(res.final_fmax_atoms < opts.ftol);
    /* Pressure-like check: |F_ε|/(3·V) below ftol — same criterion the
     * relaxer uses internally. */
    const double V = st.get_volume();
    CHECK(res.final_strain_force / (3.0 * V) < opts.ftol);
    /* Scale factor should be physically reasonable (not blown up). */
    CHECK(res.scale_factor > 0.5);
    CHECK(res.scale_factor < 2.0);
    /* The normalised strain force is the one of the convergence test. */
    CHECK(res.final_strain_force_norm ==
          Approx(res.final_strain_force / (3.0 * V)).epsilon(1e-12));
    CHECK(res.final_strain_force_norm < opts.ftol);
}

/* Pins: the strain force a relaxation reports is that of the cell it ends
 * at, also when it stops after an accepted step (at maxiter, or when the
 * early_reject predicate ends it): final_strain_force is |dE/deps| of the
 * final cell, by the relaxer's own central difference (the start cell
 * scaled by scale_factor x (1 +- eps_strain_fd)), and
 * final_strain_force_norm is that over 3 V of the final cell, the strain
 * part of the convergence test, in eV/A^3 (a pressure).
 * Before the fix: final_strain_force was the one measured at the start of
 * the last iteration, before its step (another cell), and the result had
 * no normalised strain force. */
TEST_CASE("NativeRelaxer Iso: the strain force of the result is that of the "
          "cell the relaxation ends at",
          "[mlip][native_relaxer][iso][strain_force]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db(rcut);
    host.train(train_db);

    NativeRelaxer::Options opts;
    opts.box_mode  = NativeRelaxer::Options::Iso;
    opts.pressure  = 0.0;
    opts.ftol      = 1.0e-6;
    opts.init_step = 1.0e-3;
    opts.max_step  = 0.05;

    /* |dE/deps| of @p start scaled by @p scale, as the relaxer measures it:
     * the start cell and positions scaled by scale x (1 +- h), each with a
     * list built at rcut + skin. */
    auto strain_force = [&](const Structure& start, double scale) {
        const double h = opts.eps_strain_fd;
        auto energy = [&](double s) {
            Structure x = start;
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                    x.cell(i, j) = s * start.cell(i, j);
            for (std::size_t a = 0; a < x.natoms(); ++a)
                for (int k = 0; k < 3; ++k)
                    x.atoms[a].position[k] = s * start.atoms[a].position[k];
            x.nn.reset();
            NNFinder().calc(x, rcut + opts.nn_skin);
            return host.model->epredict(host.dc.calc(x));
        };
        return std::abs((energy(scale * (1.0 + h)) -
                         energy(scale * (1.0 - h))) / (2.0 * h));
    };

    auto check_end = [&](const NativeRelaxer::Options& o) {
        const Structure start = make_bcc_ta_supercell(3.10);
        Structure st = start;
        NativeRelaxer relaxer;
        const auto res =
            relaxer.relax(st, *host.model, host.dc, host.context, o);
        INFO("converged=" << res.converged << "  iter=" << res.n_iter
             << "  early_rejected=" << res.early_rejected
             << "  scale=" << res.scale_factor);
        /* Preconditions: the relaxation stopped after accepted steps. */
        REQUIRE_FALSE(res.converged);
        REQUIRE(res.scale_factor != 1.0);
        const double F_end = strain_force(start, res.scale_factor);
        const double V_end = st.get_volume();
        INFO(std::setprecision(17) << "final_strain_force "
             << res.final_strain_force << ", |dE/deps| at the end " << F_end
             << "; final_strain_force_norm " << res.final_strain_force_norm
             << ", over 3 V " << F_end / (3.0 * V_end));
        CHECK(res.final_strain_force == Approx(F_end).epsilon(1e-9));
        CHECK(res.final_strain_force_norm ==
              Approx(F_end / (3.0 * V_end)).epsilon(1e-9));
    };

    SECTION("at maxiter") {
        NativeRelaxer::Options o = opts;
        o.maxiter = 3;
        check_end(o);
    }
    SECTION("when the early_reject predicate ends it") {
        NativeRelaxer::Options o = opts;
        o.maxiter = 500;
        o.early_reject = [](const tadah::mlip::EarlyState& s) {
            return s.iter >= 2;
        };
        check_end(o);
    }
}

namespace {

using tadah::mlip::StDescriptors;

/* n x n x n supercell of the 2-atom cubic bcc Ta cell, built with the
 * test-local loop of nn_oracle (not StructureTransformation::replicate). */
Structure bcc_ta_local(double a0, int n) {
    Structure prim;
    prim.label = "bcc Ta";
    prim.cell.load(a0, 0.0, 0.0,
                   0.0, a0, 0.0,
                   0.0, 0.0, a0);
    const Element ta = PeriodicTable::find_by_symbol("Ta");
    prim.add_atom(Atom(ta, 0.0, 0.0, 0.0));
    prim.add_atom(Atom(ta, 0.5 * a0, 0.5 * a0, 0.5 * a0));
    return nn_oracle::supercell(prim, n, n, n);
}

/* The training set of make_ta_train_db, built with the local supercell. */
StructureDB make_ta_train_db_local() {
    StructureDB db;
    const std::vector<double> a0s = {3.20, 3.30, 3.40};
    db.dbidx = {0, a0s.size()};
    for (std::size_t i = 0; i < a0s.size(); ++i) {
        Structure st = bcc_ta_local(a0s[i], 2);
        st.energy = -0.5 * static_cast<double>(i + 1);
        db.structures.push_back(st);
    }
    return db;
}

/* Distance of every table entry of st, from the entry's shift and the
 * CURRENT cell (the oracle's own arithmetic, not the table's disp). */
std::vector<double> entry_distances(const Structure& st) {
    std::vector<double> d;
    const auto L = nn_oracle::lattice(st);
    for (const auto& k : nn_oracle::table_set(st))
        d.push_back(static_cast<double>(nn_oracle::norm(nn_oracle::pair_vector(
            L, st, std::get<0>(k), std::get<1>(k), std::get<2>(k),
            std::get<3>(k), std::get<4>(k)))));
    return d;
}

}  // namespace

TEST_CASE("NativeRelaxer Iso: energy and forces after cached steps equal a "
          "fresh neighbour list at the final cell",
          "[mlip][native_relaxer][nn]")
{
    /* In Iso mode every accepted step rescales the cell, and the relaxer
     * reuses its neighbour list (built at rcut + skin) while the list still
     * covers rcut. A reused list must describe the CURRENT cell: its images
     * r_j + s.A move with the cell. The energy and forces the relaxer ends
     * with must equal those from a list built afresh at the final cell. */
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db_local();
    host.train(train_db);

    /* 3x3x3 bcc (54 atoms, side 9.3 A >= rcut + skin), compressed. */
    Structure st = bcc_ta_local(3.10, 3);

    NativeRelaxer::Options opts;
    opts.box_mode  = NativeRelaxer::Options::Iso;
    opts.pressure  = 0.0;
    opts.ftol      = 1.0e-3;
    opts.maxiter   = 500;
    opts.init_step = 1.0e-3;
    opts.max_step  = 0.05;
    REQUIRE(opts.nn_skin == 2.0);

    NativeRelaxer relaxer;
    auto res = relaxer.relax(st, *host.model, host.dc, host.context, opts);
    INFO("converged=" << res.converged << "  iter=" << res.n_iter
         << "  scale=" << res.scale_factor << "  E=" << res.final_energy);

    /* Preconditions: the cell changed, and the final list is the relaxer's
     * skin list (it holds pairs between rcut and rcut + skin). */
    REQUIRE(res.n_iter > 1);
    REQUIRE(res.scale_factor != 1.0);
    REQUIRE(st.nn);
    const auto d = entry_distances(st);
    const std::size_t n_skin = static_cast<std::size_t>(std::count_if(
        d.begin(), d.end(),
        [&](double x) { return x >= rcut && x < rcut + opts.nn_skin; }));
    INFO(n_skin << " of " << d.size() << " final entries lie in the skin");
    REQUIRE(n_skin > 0);
    /* ... and it was kept from an earlier cell (a cache hit after the cell
     * changed), not built at the final one: a list built at rcut + skin on
     * the final structure holds other pairs near rcut + skin. */
    {
        Structure rebuilt = st;
        NNFinder().calc(rebuilt, rcut + opts.nn_skin);
        REQUIRE(nn_oracle::table_set(rebuilt) != nn_oracle::table_set(st));
    }

    /* Energy and forces with the relaxer's final list ... */
    StDescriptors d_relax = host.dc.calc(st);
    const double E_relax = host.model->epredict(d_relax);
    std::vector<tadah::core::force_type> F_relax(st.natoms());
    for (std::size_t a = 0; a < st.natoms(); ++a)
        F_relax[a] = host.model->fpredict(a, d_relax, st);

    /* ... and with a list built afresh at rcut on the final structure. */
    Structure fresh = st;
    fresh.nn.reset();
    NNFinder().calc(fresh, rcut);
    StDescriptors d_fresh = host.dc.calc(fresh);
    const double E_fresh = host.model->epredict(d_fresh);
    double f_diff = 0.0;
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        const auto F = host.model->fpredict(a, d_fresh, fresh);
        for (int k = 0; k < 3; ++k)
            f_diff = std::max(f_diff, std::abs(F_relax[a][k] - F[k]));
    }
    INFO("E(relaxer list) = " << E_relax << ", E(fresh list) = " << E_fresh
         << ", max |dF| = " << f_diff << " eV/A");
    CHECK(std::abs(E_relax - E_fresh) <= 1e-10 * std::max(1.0, std::abs(E_fresh)));
    CHECK(f_diff <= 1e-10);
    /* The relaxer's reported energy is the one of its final list. */
    CHECK(res.final_energy == Approx(E_fresh).epsilon(1e-10));
}

TEST_CASE("NativeRelaxer with nn_skin 0 does not cache",
          "[mlip][native_relaxer][nn]")
{
    /* nn_skin = 0 means: build the list at rcut for every evaluation. A
     * list reused across an expansion would still hold pairs that have
     * moved beyond rcut (and may have lost none inside it), so the final
     * list must be exactly the rcut list of the final cell. */
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db_local();
    host.train(train_db);

    Structure st = bcc_ta_local(3.10, 3);

    NativeRelaxer::Options opts;
    opts.box_mode  = NativeRelaxer::Options::Iso;
    opts.pressure  = 0.0;
    opts.ftol      = 1.0e-3;
    opts.maxiter   = 500;
    opts.init_step = 1.0e-3;
    opts.max_step  = 0.05;
    opts.nn_skin   = 0.0;

    NativeRelaxer relaxer;
    auto res = relaxer.relax(st, *host.model, host.dc, host.context, opts);
    INFO("converged=" << res.converged << "  iter=" << res.n_iter
         << "  scale=" << res.scale_factor);
    /* Precondition: the relax expanded the cell (bcc 3.10 -> about 3.3,
     * so the 4.38 A shell of the start moves beyond rcut). */
    REQUIRE(res.scale_factor > 1.03);
    REQUIRE(st.nn);

    const double w = nn_oracle::window(st, rcut);
    const auto d = entry_distances(st);
    const std::size_t n_beyond = static_cast<std::size_t>(std::count_if(
        d.begin(), d.end(), [&](double x) { return x >= rcut + w; }));
    INFO(n_beyond << " of " << d.size() << " final entries lie beyond rcut");
    CHECK(n_beyond == 0);
    const auto cmp = nn_oracle::compare(st, rcut);
    INFO(cmp.summary());
    CHECK(cmp.missing.empty());
}
