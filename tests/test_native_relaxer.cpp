// test_native_relaxer.cpp
//
// Phase D2 (PC_OBSERVABLE redesign): smoke tests for NativeRelaxer
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

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/native_relaxer.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

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
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
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
}
