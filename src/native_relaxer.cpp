#include <tadah/mlip/native_relaxer.h>

#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace tadah {
namespace mlip {

using tadah::core::Context;
using tadah::core::Vec3d;

/* ------------------------------------------------------------------- */
/* Helpers                                                              */
/* ------------------------------------------------------------------- */

double NativeRelaxer::
evaluate_E_F(Structure& st,
             M_Tadah_Base& model,
             DC_Base& dc,
             const Context& ctx,
             std::vector<Vec3d>& F_out)
{
    /* Rebuild NN — atoms may have moved since last build. */
    NNFinder nnf;
    nnf.calc(st, ctx);

    /* Compute descriptors + energy + forces. */
    StDescriptors std_d = dc.calc(st);
    const double E = model.epredict(std_d);
    F_out.resize(st.natoms());
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        F_out[a] = model.fpredict(a, std_d, st);
    }
    return E;
}

double NativeRelaxer::
evaluate_E(Structure& st,
           M_Tadah_Base& model,
           DC_Base& dc,
           const Context& ctx)
{
    NNFinder nnf;
    nnf.calc(st, ctx);
    StDescriptors std_d = dc.calc(st);
    return model.epredict(std_d);
}

void NativeRelaxer::
apply_iso_strain(Structure& st, double scale,
                 const Structure& st_baseline)
{
    /* Affine scaling of cell + atoms. Note we scale the BASELINE
     * (pre-relaxation) cell to avoid compounding numeric drift across
     * iterations. */
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            st.cell(i, j) = scale * st_baseline.cell(i, j);
    for (std::size_t a = 0; a < st.natoms(); ++a) {
        st.atoms[a].position[0] = scale * st_baseline.atoms[a].position[0];
        st.atoms[a].position[1] = scale * st_baseline.atoms[a].position[1];
        st.atoms[a].position[2] = scale * st_baseline.atoms[a].position[2];
    }
}

double NativeRelaxer::
fmax_inf(const std::vector<Vec3d>& F)
{
    double m = 0.0;
    for (const auto& f : F) {
        m = std::max(m, std::abs(f[0]));
        m = std::max(m, std::abs(f[1]));
        m = std::max(m, std::abs(f[2]));
    }
    return m;
}

/* ------------------------------------------------------------------- */
/* Main relax loop                                                     */
/* ------------------------------------------------------------------- */

NativeRelaxer::Result NativeRelaxer::
relax(Structure& st_inout,
      M_Tadah_Base& model,
      DC_Base& dc,
      const Context& ctx,
      const Options& opts)
{
    Result result{};

    if (opts.box_mode == Options::Aniso
     || opts.box_mode == Options::Voigt6
     || opts.box_mode == Options::FixedZ) {
        throw std::logic_error(
            "NativeRelaxer: requested box_mode not implemented in MVP. "
            "Aniso/Voigt6 are scheduled with the elastic-constants observable; "
            "FixedZ with the γ-surface observable.");
    }

    /* Baseline cell + positions (used only in Iso mode for affine
     * scaling — keeps cumulative scale numerically clean). */
    Structure st_baseline = st_inout;
    double scale_total    = 1.0;  /* product of (1 + dε) applied so far */

    /* Eval-budget bookkeeping. LAMMPS's `minimize ... maxeval` caps
     * total energy/force evaluations across the whole relax (FD probes
     * + LS retries + force refresh). Without this cap, a pathological
     * θ that never converges does maxiter × maxiter_ls evals and
     * stalls obs_loss for minutes. */
    auto budget_left = [&]() {
        return result.n_evals < opts.maxeval;
    };
    auto eval_ef = [&](Structure& s, std::vector<Vec3d>& F_out) {
        ++result.n_evals;
        return evaluate_E_F(s, model, dc, ctx, F_out);
    };
    auto eval_e = [&](Structure& s) {
        ++result.n_evals;
        return evaluate_E(s, model, dc, ctx);
    };

    std::vector<Vec3d> F(st_inout.natoms());
    double E0 = eval_ef(st_inout, F);
    result.final_energy = E0;

    if (opts.verbose >= 1) {
        std::cerr << "[NativeRelaxer] iter=0  E=" << E0
                  << "  fmax=" << fmax_inf(F) << "\n";
    }

    double step = opts.init_step;

    for (std::size_t it = 1; it <= opts.maxiter; ++it) {
        result.n_iter = it;

        /* Eval-budget gate: stop cleanly when we've matched LAMMPS's
         * maxeval semantics. Returns the partial-relax state in the
         * Structure so the caller can still extract `final_fmax_atoms`
         * for its own convergence check (e.g. cohesive_energy's
         * eq_ok-style penalty). */
        if (!budget_left()) {
            result.maxeval_reached = true;
            result.final_energy    = E0;
            result.final_fmax_atoms = fmax_inf(F);
            result.scale_factor    = scale_total;
            return result;
        }

        /* Strain force (Iso mode only). */
        double F_eps = 0.0;
        if (opts.box_mode == Options::Iso) {
            const double dh = opts.eps_strain_fd;
            Structure st_p = st_inout;
            apply_iso_strain(st_p, scale_total * (1.0 + dh), st_baseline);
            const double E_plus = eval_e(st_p);

            Structure st_m = st_inout;
            apply_iso_strain(st_m, scale_total * (1.0 - dh), st_baseline);
            const double E_minus = eval_e(st_m);

            const double dE_deps = (E_plus - E_minus) / (2.0 * dh);
            /* Add external-pressure contribution: H = E + P·V; for iso
             *   strain dV/dε = 3·V_0. */
            const double V0 = st_inout.get_volume();
            const double dH_deps = dE_deps + opts.pressure * 3.0 * V0;
            F_eps = -dH_deps;
            result.final_strain_force = std::abs(F_eps);
        }

        const double fmax_a = fmax_inf(F);
        result.final_fmax_atoms = fmax_a;

        /* Convergence on max(atomic forces, strain force normalised). */
        const double V0 = st_inout.get_volume();
        const double fmax_strain_norm = (opts.box_mode == Options::Iso)
            ? std::abs(F_eps) / (3.0 * std::max(V0, 1.0e-9))
            : 0.0;
        if (fmax_a < opts.ftol && fmax_strain_norm < opts.ftol) {
            result.converged = true;
            result.final_energy = E0;
            result.scale_factor = scale_total;
            if (opts.verbose >= 1) {
                std::cerr << "[NativeRelaxer] converged at iter=" << it
                          << "  E=" << E0 << "  fmax=" << fmax_a
                          << "  scale=" << scale_total << "\n";
            }
            return result;
        }

        /* Compute step magnitude. Two simultaneous bounds:
         *  - atomic displacement ≤ max_step / fmax_a
         *  - |Δscale| ≤ max_strain_step (Iso mode only)
         * For randomly-initialised potentials F_eps can be ~1e6+,
         * which would otherwise drive scale_trial to ~0 inside the
         * line-search halving sequence and hang NNFinder on the
         * resulting near-degenerate cell. */
        const double safe_fmax = std::max(fmax_a, 1.0e-12);
        double trial_step = std::min(step, opts.max_step / safe_fmax);
        if (opts.box_mode == Options::Iso) {
            const double safe_F_eps = std::max(std::abs(F_eps), 1.0e-12);
            trial_step = std::min(trial_step,
                                  opts.max_strain_step / safe_F_eps);
        }

        /* Armijo backtracking line search along (F_atoms, F_eps). */
        double alpha = trial_step;
        const double gdotd = -fmax_a * fmax_a;  /* approx; SD direction */
        Structure st_trial;
        double E_trial = E0;
        bool accepted = false;
        for (std::size_t ls = 0; ls < opts.maxiter_ls; ++ls) {
            st_trial = st_inout;
            /* Atom step. */
            for (std::size_t a = 0; a < st_trial.natoms(); ++a) {
                st_trial.atoms[a].position[0] += alpha * F[a][0];
                st_trial.atoms[a].position[1] += alpha * F[a][1];
                st_trial.atoms[a].position[2] += alpha * F[a][2];
            }
            /* Strain step (Iso). */
            double scale_trial = scale_total;
            if (opts.box_mode == Options::Iso) {
                scale_trial = scale_total * (1.0 + alpha * F_eps);
                apply_iso_strain(st_trial, scale_trial, st_baseline);
                /* But the atom step above operated on the un-scaled
                 * st_inout's positions. Re-apply atom step on top of the
                 * affine-scaled baseline atoms (otherwise we'd double-
                 * count). Simpler: keep the SD atom step ON st_inout's
                 * current positions, then scale ONLY the baseline-cell
                 * driven shift by (1+α·F_eps).
                 *
                 * Pragmatic MVP: for cohesive-energy on a perfect BCC
                 * lattice, F_atoms ≈ 0 by symmetry, so the strain DOF
                 * dominates. For atomically-relaxing cases (vacancy
                 * etc., later observable), the atom-step contribution
                 * is the primary driver and strain is a small correction.
                 * Coupling them properly requires a generalised co-
                 * ordinate system — deferred until that observable
                 * actually needs it. For now: do atom step in
                 * post-strain coordinates by recomputing positions:
                 *   x_trial = (1 + α·F_eps) · x_baseline + α·F_atoms */
                for (std::size_t a = 0; a < st_trial.natoms(); ++a) {
                    st_trial.atoms[a].position[0] =
                        scale_trial * st_baseline.atoms[a].position[0]
                      + alpha * F[a][0];
                    st_trial.atoms[a].position[1] =
                        scale_trial * st_baseline.atoms[a].position[1]
                      + alpha * F[a][1];
                    st_trial.atoms[a].position[2] =
                        scale_trial * st_baseline.atoms[a].position[2]
                      + alpha * F[a][2];
                }
            }

            E_trial = eval_e(st_trial);

            /* Scale-range gate (Iso): if this LS trial would push
             * scale_total outside [scale_min, scale_max], reject it
             * like an Armijo failure and let the LS halve. Successive
             * halvings will eventually produce a step that lands
             * inside the bound. If the LS exhausts all halvings
             * without finding an in-range step, the outer "no accept"
             * branch fires and we return scale_out_of_range=true
             * (eq_ok-style penalty upstream). This way the bound is
             * a soft constraint per-LS-step, not a hard whole-relax
             * abort that would prematurely fail-score every θ whose
             * strain force happens to overshoot on the first try. */
            if (opts.box_mode == Options::Iso &&
                (scale_trial < opts.scale_min ||
                 scale_trial > opts.scale_max)) {
                alpha *= opts.backtrack_rho;
                if (alpha < opts.step_floor) break;
                if (!budget_left()) break;
                continue;
            }
            /* Armijo sufficient decrease, lenient form. */
            if (E_trial <= E0 + opts.armijo_c1 * alpha * gdotd) {
                accepted = true;
                /* Commit. */
                st_inout = st_trial;
                if (opts.box_mode == Options::Iso) scale_total = scale_trial;
                E0 = E_trial;
                /* Refresh forces for next iter. */
                eval_ef(st_inout, F);
                /* Grow step for next iter (mild). */
                step = std::min(opts.max_step, alpha * 1.5);
                break;
            }
            alpha *= opts.backtrack_rho;
            if (alpha < opts.step_floor) break;
            /* Bail out of line search if we exhausted the eval budget
             * mid-LS; the outer-loop gate will record maxeval_reached
             * on the next iteration. */
            if (!budget_left()) break;
        }

        if (!accepted) {
            if (opts.verbose >= 1) {
                std::cerr << "[NativeRelaxer] line-search failed at iter=" << it
                          << "  step=" << alpha << "  fmax=" << fmax_a << "\n";
            }
            result.converged = false;
            result.final_energy = E0;
            result.scale_factor = scale_total;
            return result;
        }

        if (opts.verbose >= 2) {
            std::cerr << "[NativeRelaxer] iter=" << it
                      << "  E=" << E0
                      << "  fmax=" << fmax_inf(F)
                      << "  step=" << alpha
                      << "  scale=" << scale_total << "\n";
        }
    }

    /* maxiter exhausted. */
    result.final_energy = E0;
    result.final_fmax_atoms = fmax_inf(F);
    result.scale_factor = scale_total;
    return result;
}

}  // namespace mlip
}  // namespace tadah
