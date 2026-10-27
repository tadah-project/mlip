#include <tadah/mlip/native_relaxer.h>

#include <tadah/mlip/nn_cache.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/st_descriptors.h>

#include <algorithm>
#include <chrono>
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
            "internal error: the relaxation takes a fixed or an isotropic box "
            "only");
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

    /* Wall-time bookkeeping (PC_OBSERVABLE analogue of LAMMPS MAXTIME).
     * Checked at iter boundaries; if exceeded, the relax exits with
     * `Result::wall_time_exceeded = true`. Disabled when max_wall_time_s
     * <= 0 (the std::chrono path stays cheap when unused). */
    const auto t0_wall = std::chrono::steady_clock::now();
    auto wall_exceeded = [&]() {
        if (opts.max_wall_time_s <= 0.0) return false;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0_wall).count();
        return elapsed > opts.max_wall_time_s;
    };

    /* NN-cache state. One NNFinder, one cache, live across all evals
     * inside this relax(). `cached` calls (st_inout's initial build and
     * its post-LS force refresh) check freshness against the prior
     * snapshot and reuse the cached NN list while it provably holds
     * every pair within rcut (nn_cache::is_fresh);
     * `uncached` calls (FD strain probes, LS trial energies) always
     * rebuild fresh since their Structure objects are transient.
     * Building NN at cutoff = rcut + skin so pairs that drift into the
     * rcut sphere are already present. With nn_skin 0 nothing is cached
     * (Options::nn_skin): every evaluation builds its own list. */
    nn_cache::State cache;
    cache.skin = opts.nn_skin;
    const bool cache_enabled = opts.nn_skin > 0.0;
    const double rcut = ctx.get<double>("RCUTMAX");
    const double nn_cutoff = rcut + cache.skin;
    NNFinder nnf;

    auto ensure_nn = [&](Structure& s, bool use_cache) {
        use_cache = use_cache && cache_enabled;
        if (use_cache && nn_cache::is_fresh(s, cache)) return;
        nnf.calc(s, nn_cutoff);
        if (use_cache) nn_cache::capture(s, cache);
    };

    auto eval_ef = [&](Structure& s, std::vector<Vec3d>& F_out,
                       bool use_cache) {
        ++result.n_evals;
        ensure_nn(s, use_cache);
        StDescriptors std_d = dc.calc(s);
        const double E = model.epredict(std_d);
        F_out.resize(s.natoms());
        for (std::size_t a = 0; a < s.natoms(); ++a) {
            F_out[a] = model.fpredict(a, std_d, s);
        }
        return E;
    };

    auto eval_e = [&](Structure& s, bool use_cache) {
        ++result.n_evals;
        ensure_nn(s, use_cache);
        StDescriptors std_d = dc.calc(s);
        return model.epredict(std_d);
    };

    std::vector<Vec3d> F(st_inout.natoms());
    double E0 = eval_ef(st_inout, F, /*use_cache=*/true);
    result.final_energy = E0;
    result.final_fmax_atoms = fmax_inf(F);

    /* Strain force (Iso mode only): F_eps = -dH/deps of the current cell,
     * the baseline scaled by scale_total, by central difference on the
     * strain (two energy evaluations), with H = E + P.V (dV/deps = 3 V).
     * record_strain keeps it in the result with |F_eps| / (3 V), the
     * strain part of the convergence test; strain_current says whether the
     * recorded value is that of st_inout, which an accepted step changes. */
    auto strain_force = [&](double scale) {
        const double dh = opts.eps_strain_fd;
        Structure st_p = st_inout;
        apply_iso_strain(st_p, scale * (1.0 + dh), st_baseline);
        const double E_plus = eval_e(st_p, /*use_cache=*/false);

        Structure st_m = st_inout;
        apply_iso_strain(st_m, scale * (1.0 - dh), st_baseline);
        const double E_minus = eval_e(st_m, /*use_cache=*/false);

        const double dE_deps = (E_plus - E_minus) / (2.0 * dh);
        const double V0 = st_inout.get_volume();
        const double dH_deps = dE_deps + opts.pressure * 3.0 * V0;
        return -dH_deps;
    };
    bool strain_current = true;
    auto record_strain = [&](double F_eps) {
        const double V0 = st_inout.get_volume();
        result.final_strain_force = std::abs(F_eps);
        result.final_strain_force_norm =
            std::abs(F_eps) / (3.0 * std::max(V0, 1.0e-9));
        strain_current = true;
    };
    /* At an exit after an accepted step, the strain force of the cell the
     * relaxation ends at (the last one measured was that of the cell
     * before the step). */
    auto finish_strain = [&]() {
        if (opts.box_mode == Options::Iso && !strain_current)
            record_strain(strain_force(scale_total));
    };

    if (opts.verbose >= 1) {
        std::cerr << "relaxation: iteration 0, E=" << E0
                  << ", fmax=" << fmax_inf(F) << "\n";
    }

    /* Iter-0 convergence check. If the initial state is already below
     * the tolerances (ftol for the atoms, ptol for the cell), declare
     * convergence without entering the iter loop. Handles two real cases:
     *  - caller asks for maxiter=0 ("don't relax, just evaluate") —
     *    test_cohesive_energy_observable.cpp:207 relies on this for
     *    FD-vs-analytical dO/dW parity on a fixed cell;
     *  - perfect BCC under Fixed/Iso where F=0 and stress=0 by symmetry;
     *    previously the relaxer needed ≥1 iter to detect convergence.
     *
     * For Iso mode we need the strain force F_ε via a one-time FD probe
     * before we can rule on convergence (the in-loop test always checks
     * both fmax_atoms AND fmax_strain_norm). Cost: 2 extra E evals on
     * iter 0; amortised against the maxiter=5000 budget this is noise. */
    {
        if (opts.box_mode == Options::Iso) record_strain(strain_force(1.0));
        const double fmax_strain_norm = result.final_strain_force_norm;
        if (result.final_fmax_atoms < opts.ftol && fmax_strain_norm < opts.ptol) {
            result.converged = true;
            result.scale_factor = 1.0;
            return result;
        }
    }

    double step = opts.init_step;

    /* Track the previous accepted energy for the etol convergence test.
     * Updated only when an LS step is accepted (E0 reassigned). */
    double E_prev = E0;

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
            finish_strain();
            return result;
        }
        if (wall_exceeded()) {
            result.wall_time_exceeded = true;
            result.final_energy       = E0;
            result.final_fmax_atoms   = fmax_inf(F);
            result.scale_factor       = scale_total;
            finish_strain();
            return result;
        }

        /* Strain force (Iso mode only). */
        double F_eps = 0.0;
        if (opts.box_mode == Options::Iso) {
            F_eps = strain_force(scale_total);
            record_strain(F_eps);
        }

        const double fmax_a = fmax_inf(F);
        result.final_fmax_atoms = fmax_a;

        /* Convergence: the atomic forces below ftol and the strain force
         * normalised, the cell's pressure, below ptol. */
        const double fmax_strain_norm = (opts.box_mode == Options::Iso)
            ? result.final_strain_force_norm
            : 0.0;
        if (fmax_a < opts.ftol && fmax_strain_norm < opts.ptol) {
            result.converged = true;
            result.final_energy = E0;
            result.scale_factor = scale_total;
            if (opts.verbose >= 1) {
                std::cerr << "relaxation: converged at iteration " << it
                          << ", E=" << E0 << ", fmax=" << fmax_a
                          << ", scale=" << scale_total << "\n";
            }
            return result;
        }

        /* Energy-tolerance convergence (LAMMPS-style etol). Skipped when
         * etol == 0 (default). Avoids the rare "force just above ftol
         * but energy already at floating-point floor" stall mode. */
        if (opts.etol > 0.0) {
            const double dE_rel = std::abs(E0 - E_prev)
                / (std::abs(E0) + std::abs(E_prev) + 1.0e-12);
            if (dE_rel < opts.etol && it > 1) {
                result.converged = true;
                result.final_energy = E0;
                result.scale_factor = scale_total;
                if (opts.verbose >= 1) {
                    std::cerr << "relaxation: converged by the energy "
                                 "tolerance at iteration " << it
                              << ", E=" << E0 << ", dE_rel=" << dE_rel << "\n";
                }
                return result;
            }
        }

        /* Compute step magnitude. Two simultaneous bounds:
         *  - atomic displacement ≤ max_step / fmax_a
         *  - |Δscale| ≤ max_strain_step (Iso mode only)
         * For randomly-initialised potentials F_eps can be ~1e6+,
         * which would otherwise drive scale_trial to ~0 inside the
         * line-search halving sequence and make NNFinder refuse the
         * resulting near-degenerate cell (its image-cell limit). */
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
        /* Track whether at least one LS trial was rejected by the soft
         * scale-range gate; used to set `result.scale_out_of_range=true`
         * if the LS exhausts. Previously the flag was declared on Result
         * but never assigned — a latent bug that cohesive_energy reads
         * (cohesive_energy.cpp:120). Fixed here as side cleanup. */
        bool any_scale_violation_in_ls = false;
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

            E_trial = eval_e(st_trial, /*use_cache=*/false);

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
                any_scale_violation_in_ls = true;
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
                strain_current = false;
                if (opts.box_mode == Options::Iso) scale_total = scale_trial;
                E_prev = E0;       /* snapshot for next iter's etol test */
                E0 = E_trial;
                /* Refresh forces for next iter. The new st_inout is at
                 * the same positions as the just-evaluated st_trial, so
                 * the NN list may still be valid (small cumulative drift
                 * vs the cached snapshot). Use the cached path: cache
                 * hit skips the rebuild entirely; miss rebuilds + updates
                 * the snapshot for next iter. */
                eval_ef(st_inout, F, /*use_cache=*/true);
                /* Grow step for next iter (mild). */
                step = std::min(opts.max_step, alpha * 1.5);
                break;
            }
            /* Armijo failure path. */
            alpha *= opts.backtrack_rho;
            if (alpha < opts.step_floor) break;
            /* Bail out of line search if we exhausted the eval budget
             * mid-LS; the outer-loop gate will record maxeval_reached
             * on the next iteration. */
            if (!budget_left()) break;
        }

        if (!accepted) {
            if (opts.verbose >= 1) {
                std::cerr << "relaxation: the line search failed at iteration "
                          << it << ", step=" << alpha << ", fmax=" << fmax_a
                          << "\n";
            }
            result.converged = false;
            result.final_energy = E0;
            result.scale_factor = scale_total;
            /* LAMMPS-style scale_out_of_range exit: if the LS exhausted
             * because every trial landed outside [scale_min, scale_max],
             * tell the caller so it can short-circuit (cohesive_energy
             * reads this flag at HPO/src/cohesive_energy.cpp:120). */
            if (any_scale_violation_in_ls) {
                result.scale_out_of_range = true;
            }
            return result;
        }

        /* Early-bail predicate hook. Called once per accepted step with
         * the post-step state; observable decides whether to keep going
         * or abort (e.g. cohesive_energy bails when scale_total drives
         * a0 outside a margin-widened hot-var range, avoiding the rest
         * of the maxeval budget on a θ that will be post-hoc rejected
         * anyway). */
        if (opts.early_reject) {
            EarlyState state{scale_total, fmax_inf(F),
                              std::abs(F_eps), it, result.n_evals};
            if (opts.early_reject(state)) {
                result.early_rejected   = true;
                result.final_energy     = E0;
                result.final_fmax_atoms = fmax_inf(F);
                result.scale_factor     = scale_total;
                finish_strain();
                return result;
            }
        }

        if (opts.verbose >= 2) {
            std::cerr << "relaxation: iteration " << it
                      << ", E=" << E0
                      << ", fmax=" << fmax_inf(F)
                      << ", step=" << alpha
                      << ", scale=" << scale_total << "\n";
        }
    }

    /* maxiter exhausted. */
    result.final_energy = E0;
    result.final_fmax_atoms = fmax_inf(F);
    result.scale_factor = scale_total;
    finish_strain();
    return result;
}

}  // namespace mlip
}  // namespace tadah
