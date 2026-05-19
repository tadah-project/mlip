#ifndef TADAH_MLIP_NATIVE_RELAXER_H
#define TADAH_MLIP_NATIVE_RELAXER_H

#include <tadah/core/context.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/descriptors_calc_base.h>

#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace tadah {
namespace mlip {

/** \brief Snapshot of relaxer state passed to user-supplied early-bail
 *         predicates. See `NativeRelaxer::Options::early_reject`.
 *
 *  Captured right after a successful line-search step (post-`E0` update,
 *  post-`scale_total` update). All fields reflect the just-accepted
 *  configuration, not the one being trialled.
 */
struct EarlyState {
    double      scale_total;   /* cumulative (1+ε), Iso only; 1.0 otherwise */
    double      fmax_atoms;    /* |F|∞ on atoms before this step (eV/Å) */
    double      fmax_strain;   /* |F_ε|, Iso only (eV/strain) */
    std::size_t iter;          /* outer-loop iteration index (1-based) */
    std::size_t n_evals;       /* total E/F evals consumed so far */
};

/** \brief In-place atomic-position + optional box-shape minimisation
 *         for the PC_OBSERVABLE pipeline.
 *
 *  Algorithm (MVP): steepest-descent on (atomic positions, optional
 *  isotropic strain DOF), with Armijo backtracking line search. The
 *  isotropic-strain force is computed by central difference on a small
 *  strain `eps_strain_fd`, since stress sign conventions across Tadah's
 *  `spredict` are non-trivial — FD here is on a single physical
 *  coordinate, NOT on n_θ or n_W, so the cost is bounded and orthogonal
 *  to the analytical-on-θ goal.
 *
 *  Box-mode strategy slots:
 *   - `Fixed`  : no box DOF. MVP.
 *   - `Iso`    : single scalar ε; cell ← (1+ε)·cell, atoms scale affinely. MVP.
 *   - `Aniso`  : reserved for elastic-constant work. Throws.
 *   - `Voigt6` : reserved. Throws.
 *   - `FixedZ` : reserved for γ-surface / surface relaxation. Throws.
 *
 *  Caller responsibilities:
 *   - The Model must be trained (weights populated).
 *   - The DC must be configured for the same Context as the Model.
 *   - The Structure's NN table is rebuilt each iteration internally.
 */
class NativeRelaxer {
public:
    struct Options {
        enum BoxMode { Fixed, Iso, Aniso, Voigt6, FixedZ };

        BoxMode      box_mode      = Fixed;
        double       pressure      = 0.0;     /* eV/Å³ (×1.602e2 = GPa) — external pressure for Iso */
        double       ftol          = 1.0e-2;  /* |F|∞ on atoms (eV/Å) */
        /* Energy convergence tolerance. When > 0 and the relative change
         * between successive accepted steps falls below this value, the
         * relax converges. Matches LAMMPS `minimize etol ftol ...`
         * (Min::energy_force in min.cpp):
         *   |E_now − E_prev| / (|E_now| + |E_prev| + 1e-12) < etol
         * Default 0 = disabled (force-only convergence). Production
         * runs targeting LAMMPS parity use 1e-15. */
        double       etol          = 0.0;
        /* Wall-time cap per relax(), in seconds. When > 0, the relax
         * exits with `Result::wall_time_exceeded = true` on the next
         * iter boundary after wall time elapses. PC_OBSERVABLE's
         * analogue of LAMMPS's `MAXTIME=1` per-HPO-eval cap: with the
         * NN cache making per-eval cost ~1 ms, an unbounded θ can burn
         * maxeval=5000 in ~5 seconds, where LAMMPS's MAXTIME would
         * have killed the subprocess at 1 s. Default 0 = disabled. */
        double       max_wall_time_s = 0.0;
        double       eps_strain_fd = 1.0e-5;  /* central-diff step for ε force (Iso) */
        double       init_step     = 0.05;    /* initial line-search trial step */
        double       max_step      = 0.5;     /* cap on per-iter atomic displacement (Å) */
        /* Maximum |Δscale| in a single line-search trial (Iso only).
         * Acts as a strain analogue of `max_step`: caps how much the
         * isotropic-strain DOF can move the cell relative to baseline
         * in one trial. Matches LAMMPS `fix box/relax iso 0.0 vmax X`
         * which caps per-iter dimension change at fraction X. Default
         * 0.001 mirrors Ecoh_fast.in (vmax 0.001). Without this bound,
         * F_eps from a wild model can drive scale_trial to ~0 within
         * the line-search halving sequence, producing a degenerate
         * cell whose NNFinder::num_shifts iterates over O(10^9+) image
         * triples and effectively hangs. */
        double       max_strain_step = 0.001;
        /* Skin distance for neighbour-list caching. NN list is built at
         * cutoff = rcut + skin so pairs that drift into the rcut sphere
         * (by up to skin/2) are already present. The list is reused
         * across iterations until any atom has moved > skin/2 since
         * the last rebuild (LAMMPS Neighbor::check_distance criterion).
         * Default 2.0 Å matches LAMMPS `neighbor 2.0 bin`. Set to 0 to
         * disable caching (rebuild every eval, original behaviour). */
        double       nn_skin       = 2.0;
        double       armijo_c1     = 1.0e-4;  /* Armijo sufficient-decrease constant */
        double       backtrack_rho = 0.5;     /* step shrink factor on line-search reject */
        double       step_floor    = 1.0e-12; /* minimum step before declaring failure */
        std::size_t  maxiter       = 5000;
        /* Hard cap on total energy/force evaluations across the whole
         * relax (FD strain probes + LS retries + accept refresh all
         * count). Mirrors LAMMPS `minimize etol ftol maxiter maxeval`
         * where maxeval is the eval budget, not the iter budget.
         * Without this cap, a pathological θ that never converges can
         * do maxiter × maxiter_ls = 250000 evals and stall the host
         * for minutes. 5000 matches LAMMPS Ecoh_fast.in. */
        std::size_t  maxeval       = 5000;
        std::size_t  maxiter_ls    = 50;
        /* Absolute bounds on scale_total (Iso mode only). Caller maps
         * its physical hot-var range onto these. E.g. cohesive_energy
         * with `--hotvar a0 2.7 3.9` and a0_current=3.3 sets
         * scale_min = 2.7/3.3, scale_max = 3.9/3.3. The relaxer bails
         * the moment an accepted step would push scale out of range,
         * preventing two pathologies:
         *  (a) cell drifts to 1-2 Å where NN_finder/dc.calc cost per
         *      eval balloons 50× (slow evals);
         *  (b) cumulative drift across HPO evals pushes a0_current_
         *      outside the hot-var range.
         * Default ±∞ → unbounded (original behaviour). */
        double       scale_min      = -std::numeric_limits<double>::infinity();
        double       scale_max      = std::numeric_limits<double>::infinity();

        /* Optional caller-supplied predicate for early termination.
         * Called once at the end of each successful line-search step
         * with the current relaxer state. Return true to abort the relax
         * with `Result::early_rejected = true`; the partial state stays
         * in the Structure so the observable can still extract whatever
         * post-hoc info it wants (e.g. `final_fmax_atoms`,
         * `scale_factor`).
         *
         * Generalizes beyond cohesive_energy: future observables supply
         * their own predicates without the relaxer needing observable-
         * specific knowledge. For cohesive_energy, the predicate bails
         * when `scale_total` would drive `a0_current * scale_total`
         * outside a margin-widened hot-var range, avoiding the
         * `maxeval=5000` burn on θ whose post-hoc hot-var rejection is
         * already inevitable.
         *
         * Nullable (default-constructed) → disabled. */
        std::function<bool(const EarlyState&)> early_reject;

        int          verbose       = 0;
    };

    struct Result {
        bool          converged       = false;
        std::size_t   n_iter          = 0;
        std::size_t   n_evals         = 0;          /* total E and E+F evaluations consumed */
        bool          maxeval_reached = false;
        bool          scale_out_of_range = false;   /* exited because LS exhausted on soft scale_min/max */
        bool          early_rejected     = false;   /* exited via Options::early_reject predicate */
        bool          wall_time_exceeded = false;   /* exited via Options::max_wall_time_s cap */
        double        final_fmax_atoms = 0.0;
        double        final_strain_force = 0.0;     /* |F_ε|, Iso only */
        double        final_energy    = 0.0;
        double        scale_factor    = 1.0;        /* cumulative (1+ε)_total, Iso only */
    };

    /** \brief Minimise the structure in place.
     *
     *  Modifies `st.atoms[i].position` and `st.cell`. Recomputes
     *  `st.nn` internally via NNFinder at the rcut taken from `ctx`.
     *
     *  Returns a Result summarising convergence. On failure
     *  (`converged=false`), the Structure is left at the last accepted
     *  step (still usable, just not at the energy minimum).
     */
    Result relax(Structure& st,
                 M_Tadah_Base& model,
                 DC_Base& dc,
                 const tadah::core::Context& ctx,
                 const Options& opts);

private:
    /* Apply iso-strain scaling: cell ← scale·cell, atoms scale affinely. */
    void apply_iso_strain(Structure& st, double scale,
                          const Structure& st_baseline);

    /* Max |F|∞ across atoms. */
    static double fmax_inf(const std::vector<tadah::core::Vec3d>& F);
};

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_NATIVE_RELAXER_H
