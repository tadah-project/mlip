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

    /* The tolerances: the largest atomic force below ftol and, for an Iso
     * cell only, the magnitude of its pressure, |F_eps| / (3V), below
     * ptol (a Fixed cell has no pressure to bound). */
    const bool iso = opts.box_mode == Options::Iso;
    auto cell_met = [&]() {
        return !iso || result.final_strain_force_norm < opts.ptol;
    };
    auto tolerances_met = [&]() {
        return result.final_fmax_atoms < opts.ftol && cell_met();
    };

    if (opts.verbose >= 1) {
        std::cerr << "relaxation: iteration 0, E=" << E0
                  << ", fmax=" << fmax_inf(F) << "\n";
    }

    /* Iter-0 convergence check. If the initial state is already below
     * the tolerances (ftol for the atoms, ptol for an Iso cell), declare
     * convergence without entering the iter loop. Handles two real cases:
     *  - caller asks for maxiter=0 ("don't relax, just evaluate") —
     *    test_cohesive_energy_observable.cpp relies on this for
     *    FD-vs-analytical dO/dW parity on a fixed cell;
     *  - perfect BCC under Fixed/Iso where F=0 and stress=0 by symmetry;
     *    previously the relaxer needed ≥1 iter to detect convergence.
     *
     * For Iso mode we need the strain force F_ε via a one-time FD probe
     * before we can rule on convergence (the in-loop test always checks
     * both fmax_atoms AND fmax_strain_norm). Cost: 2 extra E evals on
     * iter 0; amortised against the maxiter=5000 budget this is noise. */
    {
        if (iso) record_strain(strain_force(1.0));
        if (tolerances_met()) {
            result.converged = true;
            result.scale_factor = 1.0;
            return result;
        }
    }

    /* An exit after a step (maxiter, maxeval, max_wall_time_s): the cell
     * the relaxation ends at is measured, and the relaxation has converged
     * when it meets the tolerances (the step that met them was the last
     * one taken); otherwise the caller sets the flag of the cap. */
    auto finish = [&]() {
        result.final_energy     = E0;
        result.final_fmax_atoms = fmax_inf(F);
        result.scale_factor     = scale_total;
        finish_strain();
        result.converged = tolerances_met();
        if (result.converged && opts.verbose >= 1) {
            std::cerr << "relaxation: converged at iteration "
                      << result.n_iter << ", E=" << E0 << ", fmax="
                      << result.final_fmax_atoms << ", scale=" << scale_total
                      << "\n";
        }
        return result.converged;
    };

    double step = opts.init_step;

    /* Track the previous accepted energy for the etol convergence test.
     * Updated only when an LS step is accepted (E0 reassigned). */
    double E_prev = E0;

    /* The cell's search (Iso): the zero of the strain force F_eps(s) of
     * the cell scaled by s, found from the values of F_eps alone, not from
     * the energy. Near the minimum the energy a pressure P leaves,
     * V P^2 / (2B), falls below the rounding error of E long before the
     * central difference that gives F_eps loses P (the energy's error over
     * 2 eps_strain_fd), so a line search on the energy stalls where the
     * pressure is still resolved. F_eps > 0 below the minimum (the cell
     * expands), < 0 above it. Until a cell of each sign is known, a step
     * goes towards the minimum: to the zero of the secant through the
     * last two cells when F_eps decreases between them, else the steepest
     * descent step init_step * F_eps; each changes the scale by at most
     * max_strain_step. Once the zero is bracketed, a step is the false
     * position of the bracket, with the Illinois halving of the value of
     * an end kept twice in a row and a bisection when two steps have not
     * halved the bracket, so the bracket shrinks to the zero. A step below
     * step_floor (relative), or a bracket with no scale between its ends,
     * ends the relaxation unconverged: the zero is resolved, and a ptol
     * that no cell there met cannot be met. An atom step resets the
     * search (it changes F_eps(s)). */
    struct CellPoint { double scale; double force; };
    bool have_prev = false, have_lo = false, have_hi = false;
    CellPoint prev{1.0, 0.0}, lo{1.0, 0.0}, hi{1.0, 0.0};
    int last_side = 0;             /* +1: lo replaced last, -1: hi */
    double width_ref = 0.0;        /* the bracket's width two steps ago */
    int steps_unhalved = 0;
    bool last_atom_step = false;
    auto reset_cell_search = [&]() {
        have_prev = have_lo = have_hi = false;
        last_side = 0;
        steps_unhalved = 0;
    };

    for (std::size_t it = 1; it <= opts.maxiter; ++it) {
        result.n_iter = it;

        /* Eval-budget gate: stop cleanly when we've matched LAMMPS's
         * maxeval semantics. Returns the partial-relax state in the
         * Structure so the caller can still extract `final_fmax_atoms`
         * for its own convergence check (e.g. cohesive_energy's
         * eq_ok-style penalty). */
        if (!budget_left()) {
            if (!finish()) result.maxeval_reached = true;
            return result;
        }
        if (wall_exceeded()) {
            if (!finish()) result.wall_time_exceeded = true;
            return result;
        }

        /* Strain force (Iso mode only). */
        double F_eps = 0.0;
        if (iso) {
            F_eps = strain_force(scale_total);
            record_strain(F_eps);
        }

        const double fmax_a = fmax_inf(F);
        result.final_fmax_atoms = fmax_a;

        /* Convergence: the atomic forces below ftol and, for an Iso cell,
         * the strain force normalised, the cell's pressure, below ptol. */
        const bool atoms_ok = fmax_a < opts.ftol;
        const bool cell_ok = cell_met();
        if (atoms_ok && cell_ok) {
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

        /* An unconverged end at the current state (a failed step). */
        auto stop_unconverged = [&]() {
            result.converged = false;
            result.final_energy = E0;
            result.scale_factor = scale_total;
            return result;
        };

        /* The step of this iteration: the cell's (Iso) when the atoms have
         * converged, or in turn with the atoms' while neither has; the
         * atoms' otherwise. */
        const bool cell_step = iso && !cell_ok && (atoms_ok || last_atom_step);
        double step_taken = 0.0;
        if (cell_step) {
            last_atom_step = false;
            const double s = scale_total;
            if (!std::isfinite(F_eps)) {
                if (opts.verbose >= 1)
                    std::cerr << "relaxation: the strain force is not a "
                                 "number at iteration " << it << "\n";
                return stop_unconverged();
            }
            /* The bracket keeps the nearest cell of each sign on its side
             * of the zero. */
            if (F_eps > 0.0 && (!have_lo || s > lo.scale) &&
                (!have_hi || s < hi.scale)) {
                if (last_side == 1 && have_hi) hi.force *= 0.5;
                lo = {s, F_eps};
                have_lo = true;
                last_side = 1;
            } else if (F_eps < 0.0 && (!have_hi || s < hi.scale) &&
                       (!have_lo || s > lo.scale)) {
                if (last_side == -1 && have_lo) lo.force *= 0.5;
                hi = {s, F_eps};
                have_hi = true;
                last_side = -1;
            }
            double s_new = s;
            bool clamped = false;
            if (have_lo && have_hi) {
                const double width = hi.scale - lo.scale;
                if (steps_unhalved == 0 || width <= 0.5 * width_ref) {
                    width_ref = width;
                    steps_unhalved = 0;
                }
                const double mid = lo.scale + 0.5 * width;
                if (steps_unhalved >= 2) {
                    s_new = mid;
                    steps_unhalved = 0;
                    width_ref = 0.5 * width;
                } else {
                    s_new = lo.scale +
                            width * (lo.force / (lo.force - hi.force));
                    ++steps_unhalved;
                }
                if (!(s_new > lo.scale && s_new < hi.scale)) s_new = mid;
                if (!(s_new > lo.scale && s_new < hi.scale)) {
                    if (opts.verbose >= 1)
                        std::cerr << "relaxation: no cell lies between the "
                                     "two that bracket zero pressure, at "
                                     "iteration " << it << "\n";
                    return stop_unconverged();
                }
            } else {
                double rel = opts.init_step * F_eps;
                if (have_prev && prev.force != F_eps && prev.scale != s) {
                    const double slope = (F_eps - prev.force) / (s - prev.scale);
                    if (slope < 0.0) rel = (s - F_eps / slope) / s - 1.0;
                }
                rel = std::max(-opts.max_strain_step,
                               std::min(opts.max_strain_step, rel));
                s_new = s * (1.0 + rel);
                if (s_new > opts.scale_max) { s_new = opts.scale_max; clamped = true; }
                if (s_new < opts.scale_min) { s_new = opts.scale_min; clamped = true; }
            }
            if (std::abs(s_new - s) <= opts.step_floor * s) {
                if (clamped) {
                    result.scale_out_of_range = true;
                } else if (opts.verbose >= 1) {
                    std::cerr << "relaxation: the step of the cell fell "
                                 "below the floor at iteration " << it
                              << ", scale=" << s << "\n";
                }
                return stop_unconverged();
            }
            /* Commit (no test of the energy). */
            apply_iso_strain(st_inout, s_new, st_baseline);
            prev = {s, F_eps};
            have_prev = true;
            scale_total = s_new;
            strain_current = false;
            E_prev = E0;
            E0 = eval_ef(st_inout, F, /*use_cache=*/true);
            step_taken = s_new / s - 1.0;
        } else {
            last_atom_step = true;
            /* Steepest descent on the atoms, at the fixed cell, with an
             * Armijo backtracking line search on the energy; the step
             * moves no atom by more than max_step. */
            const double safe_fmax = std::max(fmax_a, 1.0e-12);
            double alpha = std::min(step, opts.max_step / safe_fmax);
            const double gdotd = -fmax_a * fmax_a;  /* approx; SD direction */
            Structure st_trial;
            bool accepted = false;
            for (std::size_t ls = 0; ls < opts.maxiter_ls; ++ls) {
                st_trial = st_inout;
                for (std::size_t a = 0; a < st_trial.natoms(); ++a) {
                    st_trial.atoms[a].position[0] += alpha * F[a][0];
                    st_trial.atoms[a].position[1] += alpha * F[a][1];
                    st_trial.atoms[a].position[2] += alpha * F[a][2];
                }
                const double E_trial = eval_e(st_trial, /*use_cache=*/false);
                /* Armijo sufficient decrease, lenient form. */
                if (E_trial <= E0 + opts.armijo_c1 * alpha * gdotd) {
                    accepted = true;
                    /* Commit. */
                    st_inout = st_trial;
                    /* The atoms' positions in the cell of scale 1, which
                     * an Iso cell scales. */
                    if (iso) {
                        for (std::size_t a = 0; a < st_inout.natoms(); ++a)
                            for (int k = 0; k < 3; ++k)
                                st_baseline.atoms[a].position[k] =
                                    st_inout.atoms[a].position[k] /
                                    scale_total;
                        reset_cell_search();
                    }
                    strain_current = false;
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
                    std::cerr << "relaxation: the line search failed at "
                                 "iteration " << it << ", step=" << alpha
                              << ", fmax=" << fmax_a << "\n";
                }
                return stop_unconverged();
            }
            step_taken = alpha;
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
                      << ", step=" << step_taken
                      << ", scale=" << scale_total << "\n";
        }
    }

    /* maxiter exhausted: converged when the final cell meets the
     * tolerances. */
    finish();
    return result;
}

}  // namespace mlip
}  // namespace tadah
