#ifndef TADAH_MLIP_NATIVE_RELAXER_H
#define TADAH_MLIP_NATIVE_RELAXER_H

#include <tadah/core/context.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/descriptors_calc_base.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace tadah {
namespace mlip {

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
        int          verbose       = 0;
    };

    struct Result {
        bool          converged       = false;
        std::size_t   n_iter          = 0;
        std::size_t   n_evals         = 0;          /* total E and E+F evaluations consumed */
        bool          maxeval_reached = false;
        bool          scale_out_of_range = false;   /* exited because scale hit scale_min/max */
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
     *  step (still useable, just not at the energy minimum).
     */
    Result relax(Structure& st,
                 M_Tadah_Base& model,
                 DC_Base& dc,
                 const tadah::core::Context& ctx,
                 const Options& opts);

private:
    /* Evaluate (E, per-atom forces) at current st using dc + model.
     * Rebuilds NN before evaluation. Returns total energy. */
    double evaluate_E_F(Structure& st,
                       M_Tadah_Base& model,
                       DC_Base& dc,
                       const tadah::core::Context& ctx,
                       std::vector<tadah::core::Vec3d>& F_out);

    /* Evaluate energy only at current st. NN-rebuild included. */
    double evaluate_E(Structure& st,
                     M_Tadah_Base& model,
                     DC_Base& dc,
                     const tadah::core::Context& ctx);

    /* Apply iso-strain scaling: cell ← scale·cell, atoms scale affinely. */
    void apply_iso_strain(Structure& st, double scale,
                          const Structure& st_baseline);

    /* Max |F|∞ across atoms. */
    static double fmax_inf(const std::vector<tadah::core::Vec3d>& F);
};

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_NATIVE_RELAXER_H
