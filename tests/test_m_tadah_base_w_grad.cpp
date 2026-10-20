// test_m_tadah_base_w_grad.cpp
//
// Parity tests for the
// M_Tadah_Base::predict_dE_dW and M_Tadah_Base::vjp_force methods that
// supply ∂E/∂W and ∂(λᵀ·F)/∂W to the typed PC_OBSERVABLE pipeline.
//
// For BF_Linear / Kern_Linear (the linear-model contract):
//   ∂E/∂W              = Σ_atoms aed_a              — exact closed form.
//   ∂(λᵀF)/∂W[c]       = Σ_a Σ_k λ_a[k] · Σ_jj (−fd_ij + fd_ji)[c,k]
// Both are LINEAR in W, so the analytical formula must match a symmetric
// central-difference w.r.t. W to floating-point noise (≤ 1e-9 relative).
// Looser bounds here would mask sign / index / convention bugs.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include <cmath>
#include <random>
#include <sstream>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using tadah::core::aed_type;
using tadah::core::force_type;
using tadah::core::t_type;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::StDescriptors;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

/* Minimal D2_BP + BF_Linear context — same shape as test_hpo_analytical's
 * fixture but stripped to just the keys M_BLR + Trainer + DC_Selector
 * need. */
Context make_context(const std::vector<double>& cgrid,
                     const std::vector<double>& sgrid,
                     double rcut)
{
    Context c;
    c.add("ATOM",     "H");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(cgrid.size()));
    c.add("TYPE2B",   static_cast<int>(sgrid.size()));
    c.add("TYPE2B",   "H"); c.add("TYPE2B", "H");
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "H"); c.add("TYPEMB", "H");
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
    c.add("FORCE",    "true");          /* enable fd capture for vjp_force test */
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

/* H3 cell, similar to test_hpo_analytical. */
Structure make_h3(double offset_x) {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 1.6 + offset_x, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 2.1, 0.0, 0.0, 0.0, 0.0));
    return st;
}

StructureDB make_h3_stdb(std::size_t N, double rcut) {
    StructureDB db;
    db.dbidx = {0, N};
    db.structures.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
        db.structures.push_back(make_h3(0.05 * static_cast<double>(i)));
    NNFinder finder;
    for (std::size_t i = 0; i < N; ++i) finder.calc(db.structures[i], rcut);
    return db;
}

/* Construct a Trainer trained on a tiny H3 db so we have a populated
 * model + weights vector to perturb. Returns the Trainer plus a fresh
 * eval Structure (NN-resolved) for the parity check. */
struct Fixture {
    MLIPWorkspaceManager ws;
    Context              ctx;
    Trainer              host;
    Structure            st_eval;

    Fixture(const std::vector<double>& cgrid,
            const std::vector<double>& sgrid,
            double rcut)
        : ws(),
          ctx(make_context(cgrid, sgrid, rcut)),
          host(ctx, ws),
          st_eval(make_h3(0.11))
    {
        auto train_db = make_h3_stdb(/*N=*/4, rcut);
        for (std::size_t i = 0; i < train_db.size(); ++i)
            train_db.structures[i].energy = 0.05 * (i + 1);
        host.train(train_db);

        /* NN on the eval structure (Trainer::train only NN's the training
         * db). */
        NNFinder finder;
        finder.calc(st_eval, rcut);
    }
};

/* Helper: compute total energy of st under the current model weights via
 * the standard predict pipeline (matches predict_energies in
 * test_hpo_analytical). */
double energy_at(Trainer& host, const Structure& st) {
    StDescriptors std_d = host.dc.calc(st);
    return host.model->epredict(std_d);
}

/* Helper: compute total force per atom (Vec3d list) of st under current
 * weights. Walks the same fpredict the model uses. */
std::vector<force_type> forces_at(Trainer& host, const Structure& st) {
    StDescriptors std_d = host.dc.calc(st);
    std::vector<force_type> f(st.natoms());
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        f[a] = host.model->fpredict(a, std_d, st);
    }
    return f;
}

double lambda_dot_F(const std::vector<force_type>& F,
                    const std::vector<force_type>& lambda) {
    double s = 0.0;
    for (std::size_t a = 0; a < F.size(); ++a)
        for (std::size_t k = 0; k < 3; ++k)
            s += lambda[a][k] * F[a][k];
    return s;
}

double max_abs_diff(const t_type& a, const t_type& b) {
    REQUIRE(a.size() == b.size());
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a(i) - b(i)));
    return m;
}

double max_abs(const t_type& v) {
    double m = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i)
        m = std::max(m, std::abs(v(i)));
    return m;
}

}  // namespace

TEST_CASE("M_Tadah_Base::predict_dE_dW matches FD on BF_Linear",
          "[mlip][model][w_grad][predict_dE_dW]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    Fixture fx(cgrid, sgrid, rcut);

    /* Analytical ∂E/∂W via the new method. */
    StDescriptors std_d = fx.host.dc.calc(fx.st_eval);
    t_type dE_dW_anal;
    fx.host.model->predict_dE_dW(std_d, fx.st_eval, dE_dW_anal);

    REQUIRE(dE_dW_anal.size() > 0);
    REQUIRE(dE_dW_anal.size() == fx.host.model->get_weights().size());

    /* FD reference: central diff each W_c with step h. For BF_Linear
     * E is linear in W, so FD = analytical to floating-point noise. */
    const t_type w0 = fx.host.model->get_weights();
    const double h = 1e-5;

    t_type dE_dW_fd(w0.size());
    for (std::size_t c = 0; c < w0.size(); ++c) dE_dW_fd(c) = 0.0;

    for (std::size_t c = 0; c < w0.size(); ++c) {
        t_type wp = w0;  wp(c) += h;
        fx.host.model->set_weights(wp);
        const double Ep = energy_at(fx.host, fx.st_eval);

        t_type wm = w0;  wm(c) -= h;
        fx.host.model->set_weights(wm);
        const double Em = energy_at(fx.host, fx.st_eval);

        dE_dW_fd(c) = (Ep - Em) / (2.0 * h);
    }
    /* Restore baseline weights for cleanliness. */
    fx.host.model->set_weights(w0);

    const double diff = max_abs_diff(dE_dW_anal, dE_dW_fd);
    const double scale = std::max(max_abs(dE_dW_anal), max_abs(dE_dW_fd));
    INFO("max |Δ| = " << diff << ", scale = " << scale);

    /* For BF_Linear ∂E/∂W is descriptor-sum, independent of W; FD must
     * agree with analytical to ~h^2 · scale (Taylor) plus float noise. */
    CHECK(diff < 1e-8 + 1e-6 * scale);
}

TEST_CASE("M_Tadah_Base::vjp_force matches FD on BF_Linear",
          "[mlip][model][w_grad][vjp_force]")
{
    PeriodicTable::initialize();

    const std::vector<double> cgrid = {0.5, 1.4};
    const std::vector<double> sgrid = {3.0, 5.0};
    const double rcut = 3.0;

    Fixture fx(cgrid, sgrid, rcut);

    /* Random λ — deterministic seed for repeatability. */
    std::mt19937 rng(20260513);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<force_type> lambda(fx.st_eval.natoms());
    for (std::size_t a = 0; a < fx.st_eval.natoms(); ++a) {
        for (std::size_t k = 0; k < 3; ++k) lambda[a][k] = uni(rng);
    }

    /* Analytical ∂(λᵀF)/∂W. */
    StDescriptors std_d = fx.host.dc.calc(fx.st_eval);
    t_type vjp_anal;
    fx.host.model->vjp_force(std_d, fx.st_eval, lambda, vjp_anal);

    REQUIRE(vjp_anal.size() == fx.host.model->get_weights().size());

    /* FD reference: λᵀF is linear in W → FD matches analytical exactly
     * up to float noise. */
    const t_type w0 = fx.host.model->get_weights();
    const double h = 1e-5;

    t_type vjp_fd(w0.size());
    for (std::size_t c = 0; c < w0.size(); ++c) vjp_fd(c) = 0.0;

    for (std::size_t c = 0; c < w0.size(); ++c) {
        t_type wp = w0;  wp(c) += h;
        fx.host.model->set_weights(wp);
        const double LFp = lambda_dot_F(forces_at(fx.host, fx.st_eval), lambda);

        t_type wm = w0;  wm(c) -= h;
        fx.host.model->set_weights(wm);
        const double LFm = lambda_dot_F(forces_at(fx.host, fx.st_eval), lambda);

        vjp_fd(c) = (LFp - LFm) / (2.0 * h);
    }
    fx.host.model->set_weights(w0);

    const double diff = max_abs_diff(vjp_anal, vjp_fd);
    const double scale = std::max(max_abs(vjp_anal), max_abs(vjp_fd));
    INFO("max |Δ| = " << diff << ", scale = " << scale);

    CHECK(diff < 1e-8 + 1e-6 * scale);
}
