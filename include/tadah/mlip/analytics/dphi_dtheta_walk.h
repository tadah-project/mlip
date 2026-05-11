#ifndef TADAH_MLIP_DPHI_DTHETA_WALK_H
#define TADAH_MLIP_DPHI_DTHETA_WALK_H

/* dphi_dtheta_walk.h - structure-level dPhi/dtheta assembly for HPO
 * analytical / hybrid gradients (Phase 2A.4 follow-up, extended to DM
 * in Phase 2B (A3)).
 *
 * The MLIP design-matrix builder (DesignMatrix::build) walks atoms +
 * neighbours and routes each pair through D2_Base::calc_aed /
 * DM_Base::calc_aed to materialise the per-atom AED, then BF_Linear's
 * calc_phi_energy_row sums those AEDs into the energy row of Phi.
 *
 * The dPhi/dtheta walk mirrors that exactly, replacing calc_aed with
 * the closed-form theta-derivative hooks:
 *
 *  - D2 arm: per-pair D2_Base::calc_dDdtheta (1 call per pair, accumulates
 *    into the per-atom dAED buffer; mirrors calc_aed scale=0.5 semantics).
 *
 *  - DM arm: per-pair DM_Base::calc_drhoi_dtheta to accumulate drho_i/dtheta
 *    across neighbours, followed by per-atom DM_Base::calc_daed_dtheta to
 *    fold (rho_i, drho_i/dtheta) into the per-atom daed/dtheta. The MB
 *    code path uses calc_aed(rhoi, aed_scratch) to populate the F'(rho)
 *    section of rhoi (= 2 f rho), which calc_daed_dtheta then dot-products
 *    with drho_i/dtheta.
 *
 * For Phase 2A/2B first cut the helper covers ONLY the energy rows
 * (FORCE=false, STRESS=false).
 *
 * Symmetry-of-pair convention matches descriptors_calc.hpp:248
 * (`d2.calc_aed(..., 0.5)`): each (i, j) pair is iterated from both
 * endpoints, so each contributes 0.5 to atom i's dAED and 0.5 to
 * atom j's dAED — giving the full pair-derivative when the two
 * per-atom AEDs are summed. (DM uses scale=1 because rho_i is
 * accumulated once per atom — there is no double-counting to undo.)
 *
 * Caller convention:
 *  - `dphi_out` is row-major (n_struct × n_cols), pre-zeroed by the
 *    caller. Energy row i lives at offset i * n_cols.
 *  - `n_cols` must equal `dcs.descriptor_size()` (or the caller's
 *    cached `factor.n`); if the basis function adds further blocks
 *    (BIAS, polynomial features), those columns are left untouched
 *    here and must be filled by a basis-function-aware walker (BIAS
 *    columns: dPhi/dtheta is zero — the BIAS column is ones, no
 *    theta-dependence; polynomial features: future work).
 *  - Returns `true` iff at least one pair contributed (i.e. the d2
 *    descriptor accepted the (theta_name, arg_pos) pair). `false`
 *    means caller should route this index to needs_fd.
 *
 *  - This helper does NOT apply DesignMatrix weighting (sqrt(eweight *
 *    eweightglob)). The caller multiplies each row by its weight after
 *    the walk to match the row layout in `dm.getPhi()` post-scale.
 */

#include <tadah/core/core_types.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/descriptors/d2/d2_base.h>
#include <tadah/models/descriptors/dm/dm_base.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/functions/kernels/kern_base.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/* Per-D2 walker — runs calc_dDdtheta over every (atom, neighbour) pair
 * within d2's cutoff and accumulates into the energy row of `dphi_out`.
 *
 * Used directly for descriptor-internal theta (CGRID2B[k], SGRID2B[k])
 * and as the per-D2 inner step for cross-cutting theta (RCUT2B,
 * RCUTMB) — see assemble_dphi_train_energy_rows below.
 *
 * Pre-conditions:
 *  - `stdb(s).nn` is populated for every s (NNFinder run with cutoff
 *    at least as large as d2.get_rcut()).
 *  - `dphi_out` has size n_struct * n_cols. NOT zeroed by this helper
 *    — accumulation semantics let the caller fold contributions from
 *    multiple D2 descriptors (composite-descriptor safe).
 *
 * Return:
 *  - true  : at least one pair contributed via d2.calc_dDdtheta.
 *  - false : no pair contributed (descriptor returns false for this
 *            (theta_name, arg_pos)) — caller should route the index to
 *            needs_fd or raise under ANALYTICAL strict mode. */
inline bool
assemble_dphi_train_energy_rows_d2(const StructureDB&                  stdb,
                                   const tadah::models::D2_Base&       d2,
                                   const std::string&                  theta_name,
                                   int                                 arg_pos,
                                   double*                             dphi_out,
                                   std::size_t                         n_cols)
{
    bool any_contrib = false;

    /* Reusable per-atom buffer (sized to n_cols). */
    tadah::core::aed_type dD_buf;
    dD_buf.resize(static_cast<long>(n_cols));

    const double rcut2b    = d2.get_rcut();
    const double rcut2b_sq = rcut2b * rcut2b;

    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        if (!st.nn)
            throw std::runtime_error(
                "assemble_dphi_train_energy_rows_d2: structure " +
                std::to_string(s) + " has no NN cache; run NNFinder first.");

        /* Sum per-atom dAED/dtheta into the energy row of dPhi. With
         * BF_Linear, calc_phi_energy_row is just sum-over-atoms of
         * the per-atom AED, so dPhi_row = sum-over-atoms of dAED. */
        const std::size_t row_off = s * n_cols;
        for (std::size_t i = 0; i < st.natoms(); ++i) {
            for (std::size_t k = 0; k < n_cols; ++k) dD_buf[k] = 0.0;

            const Atom& a1 = st(i);
            const int   Zi = a1.Z;
            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const tadah::core::Vec3d& a2pos = st.nn_pos(i, jj);
                const double dx = a1.position[0] - a2pos[0];
                const double dy = a1.position[1] - a2pos[1];
                const double dz = a1.position[2] - a2pos[2];
                const double rij_sq = dx*dx + dy*dy + dz*dz;
                if (rij_sq >= rcut2b_sq) continue;
                const std::size_t neighIdx = st.nn->idx[i][jj];
                const int    Zj  = st(neighIdx).Z;
                const double rij = std::sqrt(rij_sq);

                /* scale=0.5 mirrors descriptors_calc.hpp:248 — each pair
                 * is double-counted across both endpoints, so the per-
                 * pair contribution is halved. */
                const bool ok = d2.calc_dDdtheta(
                    theta_name, arg_pos, Zi, Zj, rij, rij_sq, dD_buf, 0.5);
                if (ok) any_contrib = true;
            }

            /* Accumulate atom i's dAED into the structure's energy row. */
            for (std::size_t k = 0; k < n_cols; ++k)
                dphi_out[row_off + k] += dD_buf[k];
        }
    }
    return any_contrib;
}

/* Per-DM walker — analog of assemble_dphi_train_energy_rows_d2 for
 * many-body descriptors. Per-atom workflow:
 *
 *   1. Zero rhoi[2*rhoisize] and drhoi[2*rhoisize] scratch buffers.
 *   2. Loop over neighbours j of atom i with rij < rcutmb:
 *        - dm.calc_rho(...)         to accumulate rho_i (the standard
 *          pre-pass, same one DescriptorsCalc::calc_rho does).
 *        - dm.calc_drhoi_dtheta(...) to accumulate drho_i/dtheta over
 *          the same neighbour set, into drhoi[rfidx..].
 *   3. dm.calc_aed(rhoi, aed_scratch) to populate the F'(rho)=2 f rho
 *      section of rhoi (at indices drfidx..). aed_scratch is discarded.
 *   4. dm.calc_daed_dtheta(...) folds (rho_i, drho_i/dtheta) into
 *      daed/dtheta for atom i.
 *   5. Sum daed into the structure's energy row of dphi_out.
 *
 * Accumulation semantics let the caller fold contributions from
 * multiple DM descriptors (composite-DM safe), same as the D2 walker.
 *
 * Pre-conditions:
 *  - `stdb(s).nn` is populated for every s with cutoff >= dm.get_rcut().
 *  - `dphi_out` has size n_struct * n_cols; NOT zeroed by this helper.
 *
 * Return:
 *  - true  : at least one atom contributed via dm.calc_daed_dtheta.
 *  - false : no atom contributed (descriptor returns false for this
 *            (theta_name, arg_pos)) — caller routes to needs_fd. */
inline bool
assemble_dphi_train_energy_rows_dm(const StructureDB&                 stdb,
                                   const tadah::models::DM_Base&      dm,
                                   const std::string&                 theta_name,
                                   int                                arg_pos,
                                   double*                            dphi_out,
                                   std::size_t                        n_cols)
{
    if (dm.rhoi_size() == 0) return false;   /* DM_Dummy or inactive */

    bool any_contrib = false;

    /* Per-atom scratch buffers. Allocated once and reused. The DM
     * convention is rhoi = [rho | F'(rho)] each sized rhoisize/rhoipsize
     * (see DM_EAD::init_rhoi: resize(2*rhoisize)). */
    const std::size_t rho_total = dm.rhoi_size() + dm.rhoip_size();
    tadah::core::rho_type rhoi  (rho_total);
    tadah::core::rho_type drhoi (rho_total);
    tadah::core::aed_type aed_scratch(n_cols);
    tadah::core::aed_type daed_buf   (n_cols);

    const double rcutmb    = dm.get_rcut();
    const double rcutmb_sq = rcutmb * rcutmb;

    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        if (!st.nn)
            throw std::runtime_error(
                "assemble_dphi_train_energy_rows_dm: structure " +
                std::to_string(s) + " has no NN cache; run NNFinder first.");

        const std::size_t row_off = s * n_cols;
        for (std::size_t i = 0; i < st.natoms(); ++i) {
            rhoi.set_zero();
            drhoi.set_zero();
            aed_scratch.set_zero();
            daed_buf.set_zero();

            const Atom& a1 = st(i);
            const int   Zi = a1.Z;

            tadah::core::Vec3d delij;
            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const tadah::core::Vec3d& a2pos = st.nn_pos(i, jj);
                delij[0] = a1.position[0] - a2pos[0];
                delij[1] = a1.position[1] - a2pos[1];
                delij[2] = a1.position[2] - a2pos[2];
                const double rij_sq = delij[0]*delij[0] +
                                      delij[1]*delij[1] +
                                      delij[2]*delij[2];
                if (rij_sq >= rcutmb_sq) continue;
                const std::size_t neighIdx = st.nn->idx[i][jj];
                const int    Zj  = st(neighIdx).Z;
                const double rij = std::sqrt(rij_sq);

                /* Accumulate rho_i (the standard MB pre-pass). */
                dm.calc_rho(Zi, Zj, rij, rij_sq, delij, rhoi, /*scale=*/1.0);
                /* Accumulate drho_i/dtheta over the same neighbour set. */
                dm.calc_drhoi_dtheta(theta_name, arg_pos,
                                     Zi, Zj, rij, rij_sq, delij, drhoi,
                                     /*scale=*/1.0);
            }

            /* Populate rhoi[drfidx..] = F'(rho) = 2 f rho. The aed
             * side-output is discarded (we only want the rhoi mutation). */
            dm.calc_aed(rhoi, aed_scratch);

            /* Convert (rho, drho/dtheta) -> daed/dtheta per atom. */
            const bool ok = dm.calc_daed_dtheta(theta_name, arg_pos,
                                                rhoi, drhoi, daed_buf,
                                                /*scale=*/1.0);
            if (ok) any_contrib = true;

            /* Sum into structure energy row. */
            for (std::size_t k = 0; k < n_cols; ++k)
                dphi_out[row_off + k] += daed_buf[k];
        }
    }
    return any_contrib;
}

/* ------------------------------------------------------------------- *
 *  Kernel-arm walker — for non-linear kernel features where the
 *  design matrix is `Phi[s, b] = sum_atoms K(basis_b, aed_atom_a)`,
 *  i.e. each column is one basis vector and the row mass is summed
 *  over atoms via the kernel rather than the dot-product.
 *
 *  Two derivative branches:
 *
 *  (1) Descriptor theta (CGRID2B, SGRID2B, RCUT2B, ...): chain through
 *      the kernel using its dK/daed = kernel.derivative(b, af):
 *
 *        dPhi[s, b]/dtheta_desc
 *           = sum_atoms (dK/daed)(basis_b, aed_atom)
 *                       . (daed_atom/dtheta_desc)
 *
 *      The per-atom (aed_atom, daed_atom) snapshot is produced by the
 *      same neighbour loop as assemble_dphi_train_energy_rows_d2: each
 *      neighbour contributes to BOTH aed (via D2_Base::calc_aed) AND
 *      daed (via D2_Base::calc_dDdtheta), with scale=0.5 mirroring
 *      descriptors_calc.hpp:248.
 *
 *  (2) Kernel hyperparameter (MPARAMS): direct application of the
 *      kernel's closed-form derivative:
 *
 *        dPhi[s, b]/dtheta_kernel
 *           = sum_atoms kernel.calc_dKdtheta(theta_name, arg_pos,
 *                                            basis_b, aed_atom)
 *
 *      Descriptor theta is unused on this branch — only the per-atom
 *      aed_atom snapshot is needed.
 *
 *  Caller convention matches the D2/DM walkers: accumulation semantics
 *  (caller pre-zeros dphi_out), `n_cols` must equal `kernel.basis.cols()`,
 *  returns `true` iff at least one term contributed.
 *
 *  Phase 2B (B-HPO) groundwork — this walker is NOT yet consumed by
 *  compute_analytical_gradient_from_trainer because non-linear kernels
 *  always trigger EKM projection in M_KRR_Train::train1, and the
 *  analytical chain through EKM requires dEKM/dtheta (SVD derivative).
 *  Layer-1.5 parity tests exercise it directly against FD on a
 *  hand-crafted Phi/basis. Phase 2C plans to wire it through once the
 *  EKM derivative lands. */
inline bool
assemble_dphi_train_energy_rows_d2_kern(
    const StructureDB&                stdb,
    const tadah::models::D2_Base&     d2,
    const tadah::models::Kern_Base&   kernel,
    const std::string&                theta_name,
    int                               arg_pos,
    double*                           dphi_out,
    std::size_t                       n_cols)
{
    const tadah::core::Matrix& basis = kernel.get_basis();
    if (basis.cols() != n_cols)
        throw std::runtime_error(
            "assemble_dphi_train_energy_rows_d2_kern: n_cols does not match "
            "kernel.basis.cols() (caller must pass the kernel-feature column "
            "count, not the descriptor size).");
    if (basis.cols() == 0) return false;

    const std::size_t aed_size = static_cast<std::size_t>(basis.rows());
    const bool        is_mparams = (theta_name == "MPARAMS");

    /* Per-atom scratch buffers. aed_buf carries the per-atom AED (input
     * to kernel.derivative / kernel.calc_dKdtheta); dD_buf carries the
     * per-atom dAED/dtheta_desc (chained via dK/daed). dK_daed receives
     * the kernel-gradient vector returned by kernel.derivative. */
    tadah::core::aed_type aed_buf;  aed_buf.resize(static_cast<long>(aed_size));
    tadah::core::aed_type dD_buf;   dD_buf .resize(static_cast<long>(aed_size));

    const double rcut2b    = d2.get_rcut();
    const double rcut2b_sq = rcut2b * rcut2b;

    bool any_contrib_dD = false;
    bool any_contrib_K  = false;

    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        if (!st.nn)
            throw std::runtime_error(
                "assemble_dphi_train_energy_rows_d2_kern: structure " +
                std::to_string(s) + " has no NN cache; run NNFinder first.");

        const std::size_t row_off = s * n_cols;
        for (std::size_t i = 0; i < st.natoms(); ++i) {
            for (std::size_t k = 0; k < aed_size; ++k) {
                aed_buf[k] = 0.0;
                dD_buf [k] = 0.0;
            }

            const Atom& a1 = st(i);
            const int   Zi = a1.Z;
            bool atom_has_neighbour_contrib_dD = false;
            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const tadah::core::Vec3d& a2pos = st.nn_pos(i, jj);
                const double dx = a1.position[0] - a2pos[0];
                const double dy = a1.position[1] - a2pos[1];
                const double dz = a1.position[2] - a2pos[2];
                const double rij_sq = dx*dx + dy*dy + dz*dz;
                if (rij_sq >= rcut2b_sq) continue;
                const std::size_t neighIdx = st.nn->idx[i][jj];
                const int    Zj  = st(neighIdx).Z;
                const double rij = std::sqrt(rij_sq);

                /* scale=0.5 matches the standard double-counting
                 * convention in descriptors_calc.hpp:248. */
                d2.calc_aed(Zi, Zj, rij, rij_sq, aed_buf, 0.5);
                if (!is_mparams) {
                    const bool ok = d2.calc_dDdtheta(
                        theta_name, arg_pos, Zi, Zj, rij, rij_sq,
                        dD_buf, 0.5);
                    if (ok) atom_has_neighbour_contrib_dD = true;
                }
            }
            if (atom_has_neighbour_contrib_dD) any_contrib_dD = true;

            /* For each basis column, fold (aed_buf, dD_buf) into the
             * kernel contribution. */
            for (std::size_t b = 0; b < n_cols; ++b) {
                tadah::core::aed_type basis_col = basis.col(b);
                if (is_mparams) {
                    double dK = 0.0;
                    const bool ok = kernel.calc_dKdtheta(
                        theta_name, arg_pos, basis_col, aed_buf, dK);
                    if (ok) {
                        dphi_out[row_off + b] += dK;
                        any_contrib_K = true;
                    }
                } else {
                    if (!atom_has_neighbour_contrib_dD) continue;
                    tadah::core::aed_type dK_daed =
                        kernel.derivative(basis_col, aed_buf);
                    /* dot product: dK/daed . daed/dtheta */
                    double dot = 0.0;
                    for (std::size_t k = 0; k < aed_size; ++k)
                        dot += dD_buf[k] * dK_daed[k];
                    dphi_out[row_off + b] += dot;
                }
            }
        }
    }

    return is_mparams ? any_contrib_K : any_contrib_dD;
}

/* Per-D2 FORCE-row walker — analog of assemble_dphi_train_energy_rows_d2
 * for the 3·N force rows per structure. Mirrors descriptors_calc.hpp
 * (calc_all + the axis-expansion) followed by dm_bf_linear's
 * calc_phi_force_rows pattern:
 *
 *   For each (atom i, neighbour jj=j):
 *     1. Compute d(fd_ij)/dtheta and d(fd_ji)/dtheta via two
 *        calc_dXijdri_dtheta calls — once with (Zi, Zj) and the i-side
 *        unit direction (delij/rij), once with (Zj, Zi) and the j-side
 *        direction (-delij/rij). scale=0.5 mirrors descriptors_calc's
 *        per-pair double-counting convention.
 *     2. Expand the radial-only result to 3 axis components for each
 *        side using the unit direction vector.
 *     3. Update *only* atom i's three force rows with
 *           dPhi[force_row(s,i,k), c] -= (d(fd_ij)(c,k)
 *                                         - d(fd_ji)(c,k))
 *        for k in {0,1,2}, c in [0, n_cols).
 *
 *   When the outer loop later reaches atom j (with i as neighbour),
 *   the same pair is iterated with swapped Z values and reversed unit
 *   direction; the row update lands in atom j's force rows and equals
 *   the negative of atom i's update — Newton's third law on the
 *   gradient.
 *
 * Pre-conditions:
 *  - `stdb(s).nn` populated with cutoff >= d2.get_rcut() for every s.
 *  - `dphi_out` has space for n_aug_rows × n_cols entries (caller-
 *    allocated, NOT zeroed by this helper).
 *  - `force_row_start_per_struct[s]` is the absolute row index in
 *    dphi_out where structure s's first force row lives (atom 0,
 *    axis 0). The walker indexes row force_row_start[s] + 3*i + k.
 *  - `n_cols` matches the design matrix column count (descriptor_size
 *    for BF_Linear / Kern_Linear).
 *
 * Bias columns: forces are independent of the per-element BIAS so
 * dPhi/dtheta for those columns is zero; the walker leaves them
 * untouched (calc_dXijdri_dtheta writes only at fidx+c, c in
 * [0, descriptor_size_2b)).
 *
 * Weight scaling: NOT applied here — caller multiplies each row by
 * `sqrt(stdb(s).fweight * FWEIGHT)` after the walk to match the
 * post-scale layout in DesignMatrix::getPhi(). */
inline bool
assemble_dphi_train_force_rows_d2(
    const StructureDB&                     stdb,
    const tadah::models::D2_Base&          d2,
    const std::string&                     theta_name,
    int                                    arg_pos,
    double*                                dphi_out,
    std::size_t                            n_cols,
    const std::vector<std::size_t>&        force_row_start_per_struct)
{
    bool any_contrib = false;

    tadah::core::fd_type dfd_ij(n_cols);
    tadah::core::fd_type dfd_ji(n_cols);

    const double rcut2b    = d2.get_rcut();
    const double rcut2b_sq = rcut2b * rcut2b;

    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        if (!st.nn)
            throw std::runtime_error(
                "assemble_dphi_train_force_rows_d2: structure " +
                std::to_string(s) + " has no NN cache; run NNFinder first.");

        const std::size_t row_start_s = force_row_start_per_struct[s];

        for (std::size_t i = 0; i < st.natoms(); ++i) {
            const Atom& a1 = st(i);
            const int   Zi = a1.Z;
            const std::size_t row_i = row_start_s + 3 * i;

            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const tadah::core::Vec3d& a2pos = st.nn_pos(i, jj);
                const double dx = a1.position[0] - a2pos[0];
                const double dy = a1.position[1] - a2pos[1];
                const double dz = a1.position[2] - a2pos[2];
                const double rij_sq = dx*dx + dy*dy + dz*dz;
                if (rij_sq >= rcut2b_sq) continue;
                const std::size_t neighIdx = st.nn->idx[i][jj];
                const int    Zj  = st(neighIdx).Z;
                const double rij = std::sqrt(rij_sq);
                const double rij_inv = 1.0 / rij;

                /* Zero only the radial column (axes 1/2 are written
                 * fresh by the axis expansion below). */
                for (std::size_t n = 0; n < n_cols; ++n) {
                    dfd_ij(n, 0) = 0.0;
                    dfd_ji(n, 0) = 0.0;
                }

                /* scale=0.5 matches descriptors_calc.hpp:229. */
                const bool ok_ij = d2.calc_dXijdri_dtheta(
                    theta_name, arg_pos, Zi, Zj, rij, rij_sq, dfd_ij, 0.5);
                const bool ok_ji = d2.calc_dXijdri_dtheta(
                    theta_name, arg_pos, Zj, Zi, rij, rij_sq, dfd_ji, 0.5);
                if (!ok_ij && !ok_ji) continue;
                any_contrib = true;

                /* Axis expansion mirrors descriptors_calc.hpp:233-238.
                 * For the ij side, delij = (dx, dy, dz). For the ji
                 * side, delji = -delij. */
                for (std::size_t n = 0; n < n_cols; ++n) {
                    const double r0_ij = dfd_ij(n, 0) * rij_inv;
                    dfd_ij(n, 0) = r0_ij * dx;
                    dfd_ij(n, 1) = r0_ij * dy;
                    dfd_ij(n, 2) = r0_ij * dz;

                    const double r0_ji = dfd_ji(n, 0) * rij_inv;
                    dfd_ji(n, 0) = r0_ji * (-dx);
                    dfd_ji(n, 1) = r0_ji * (-dy);
                    dfd_ji(n, 2) = r0_ji * (-dz);
                }

                /* Update atom i's 3 force rows: subtract (dfij - dfji)
                 * to mirror dm_bf_linear::calc_phi_force_rows. */
                for (std::size_t k = 0; k < 3; ++k) {
                    const std::size_t row_off = (row_i + k) * n_cols;
                    for (std::size_t n = 0; n < n_cols; ++n) {
                        dphi_out[row_off + n] -= (dfd_ij(n, k) - dfd_ji(n, k));
                    }
                }
            }
        }
    }
    return any_contrib;
}

/* Per-DM FORCE-row walker — many-body sibling of
 * assemble_dphi_train_force_rows_d2. Closer in shape to the energy DM
 * walker (per-atom rho/drho pre-pass + calc_aed to populate F'(rho))
 * because the DM force descriptor is per-pair AND per-atom (each pair
 * needs rho_i and drho_i for atom i, AND rho_j and drho_j for atom j on
 * the symmetric ji-side).
 *
 * Per-structure workflow:
 *   1. Pre-allocate per-atom rho / drho / F' scratch (one rho_type per
 *      atom; n = natoms * rho_total slots).
 *   2. For each atom i: accumulate rho_i over neighbours (calc_rho),
 *      drho_i/dθ over the same neighbours (calc_drhoi_dtheta), then
 *      run calc_aed(rho_i, _) once to populate the F'(rho_i) tail in
 *      rho_i's scratch.
 *   3. For each (atom i, neighbour jj=j): call calc_dXijdri_dtheta
 *      twice — once with (Zi, Zj, +delij, rhoi[i], drhoi[i]) producing
 *      dfij, once with (Zj, Zi, -delij, rhoi[j], drhoi[j]) producing
 *      dfji. The walker then updates atom i's three force rows with
 *      `dphi[force_row(s,i,k), c] -= (dfij(c,k) - dfji(c,k))` —
 *      mirroring dm_bf_linear::calc_phi_force_rows.
 *
 * Returns true iff at least one call to calc_dXijdri_dtheta claimed
 * the theta.
 *
 * The dfd_ij / dfd_ji buffers are full axis-expanded (n_cols × 3),
 * unlike the D2 sibling's radial-only (n_cols × 1) layout — the MB
 * descriptor's calc_dXijdri_dtheta does the axis expansion internally
 * (poly = x^lx y^ly z^lz couples to direction, so a single radial
 * scalar can't represent the gradient). */
inline bool
assemble_dphi_train_force_rows_dm(
    const StructureDB&                     stdb,
    const tadah::models::DM_Base&          dm,
    const std::string&                     theta_name,
    int                                    arg_pos,
    double*                                dphi_out,
    std::size_t                            n_cols,
    const std::vector<std::size_t>&        force_row_start_per_struct)
{
    if (dm.rhoi_size() == 0) return false;  /* DM_Dummy or inactive */

    const std::size_t rho_total = dm.rhoi_size() + dm.rhoip_size();
    tadah::core::aed_type aed_scratch(n_cols);

    bool any_contrib = false;

    const double rcutmb    = dm.get_rcut();
    const double rcutmb_sq = rcutmb * rcutmb;

    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        if (!st.nn)
            throw std::runtime_error(
                "assemble_dphi_train_force_rows_dm: structure " +
                std::to_string(s) + " has no NN cache; run NNFinder first.");

        const std::size_t natoms = st.natoms();

        /* Per-atom rho / drho buffers. Each atom's rhoi has size
         * rho_total = rhoi_size + rhoip_size (rho part then F'(rho) tail). */
        std::vector<tadah::core::rho_type> rhoi_per_atom (natoms);
        std::vector<tadah::core::rho_type> drhoi_per_atom(natoms);
        for (std::size_t i = 0; i < natoms; ++i) {
            rhoi_per_atom [i].resize(static_cast<long>(rho_total));
            drhoi_per_atom[i].resize(static_cast<long>(rho_total));
            rhoi_per_atom [i].set_zero();
            drhoi_per_atom[i].set_zero();
        }

        /* Pre-pass: rho_i and drho_i/dθ for every atom. */
        for (std::size_t i = 0; i < natoms; ++i) {
            const Atom& a1 = st(i);
            const int   Zi = a1.Z;

            tadah::core::Vec3d delij;
            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const tadah::core::Vec3d& a2pos = st.nn_pos(i, jj);
                delij[0] = a1.position[0] - a2pos[0];
                delij[1] = a1.position[1] - a2pos[1];
                delij[2] = a1.position[2] - a2pos[2];
                const double rij_sq = delij[0]*delij[0] +
                                      delij[1]*delij[1] +
                                      delij[2]*delij[2];
                if (rij_sq >= rcutmb_sq) continue;
                const std::size_t neighIdx = st.nn->idx[i][jj];
                const int    Zj  = st(neighIdx).Z;
                const double rij = std::sqrt(rij_sq);

                dm.calc_rho(Zi, Zj, rij, rij_sq, delij,
                            rhoi_per_atom[i], /*scale=*/1.0);
                dm.calc_drhoi_dtheta(theta_name, arg_pos,
                                     Zi, Zj, rij, rij_sq, delij,
                                     drhoi_per_atom[i], /*scale=*/1.0);
            }
            /* Populate F'(rho) tail in rhoi_per_atom[i]. */
            aed_scratch.set_zero();
            dm.calc_aed(rhoi_per_atom[i], aed_scratch);
        }

        /* Per-pair force-row update. */
        const std::size_t row_start_s = force_row_start_per_struct[s];

        /* Per-pair scratch (axis-expanded). */
        tadah::core::fd_type dfd_ij(n_cols);
        tadah::core::fd_type dfd_ji(n_cols);

        for (std::size_t i = 0; i < natoms; ++i) {
            const Atom& a1 = st(i);
            const int   Zi = a1.Z;
            const std::size_t row_i = row_start_s + 3 * i;

            tadah::core::Vec3d delij;
            for (std::size_t jj = 0; jj < st.nn_size(i); ++jj) {
                const tadah::core::Vec3d& a2pos = st.nn_pos(i, jj);
                delij[0] = a1.position[0] - a2pos[0];
                delij[1] = a1.position[1] - a2pos[1];
                delij[2] = a1.position[2] - a2pos[2];
                const double rij_sq = delij[0]*delij[0] +
                                      delij[1]*delij[1] +
                                      delij[2]*delij[2];
                if (rij_sq >= rcutmb_sq) continue;
                const std::size_t neighIdx = st.nn->idx[i][jj];
                const int    Zj  = st(neighIdx).Z;
                const double rij = std::sqrt(rij_sq);

                /* Zero per-pair scratch fully (all 3 axes). */
                for (std::size_t n = 0; n < n_cols; ++n) {
                    for (std::size_t k = 0; k < 3; ++k) {
                        dfd_ij(n, k) = 0.0;
                        dfd_ji(n, k) = 0.0;
                    }
                }

                const bool ok_ij = dm.calc_dXijdri_dtheta(
                    theta_name, arg_pos, Zi, Zj, rij, rij_sq,
                    delij, rhoi_per_atom[i], drhoi_per_atom[i],
                    dfd_ij, /*scale=*/1.0);

                tadah::core::Vec3d delji = delij;
                delji[0] = -delji[0];
                delji[1] = -delji[1];
                delji[2] = -delji[2];
                const bool ok_ji = dm.calc_dXijdri_dtheta(
                    theta_name, arg_pos, Zj, Zi, rij, rij_sq,
                    delji, rhoi_per_atom[neighIdx], drhoi_per_atom[neighIdx],
                    dfd_ji, /*scale=*/1.0);
                if (!ok_ij && !ok_ji) continue;
                any_contrib = true;

                /* Mirrors dm_bf_linear::calc_phi_force_rows: subtract
                 * (dfij - dfji) into atom i's three force rows. */
                for (std::size_t k = 0; k < 3; ++k) {
                    const std::size_t row_off = (row_i + k) * n_cols;
                    for (std::size_t n = 0; n < n_cols; ++n) {
                        dphi_out[row_off + n] -= (dfd_ij(n, k) - dfd_ji(n, k));
                    }
                }
            }
        }
    }
    return any_contrib;
}

/* DC_Selector-level FORCE-row walker. Iterates the active D2 + DM arms
 * in sequence, accumulating into the same dphi_out row. RCUT2B / RCUTMB
 * keys are descriptor-specific so a given (theta_name, arg_pos) is
 * owned by at most one arm. Mirrors assemble_dphi_train_energy_rows. */
inline bool
assemble_dphi_train_force_rows(
    const StructureDB&                     stdb,
    const tadah::models::DC_Selector&      dcs,
    const std::string&                     theta_name,
    int                                    arg_pos,
    double*                                dphi_out,
    std::size_t                            n_cols,
    const std::vector<std::size_t>&        force_row_start_per_struct)
{
    if (!dcs.d2b)
        throw std::runtime_error(
            "assemble_dphi_train_force_rows: DC_Selector has no D2 descriptor; "
            "DC_Selector::init always allocates a D2_Dummy fallback — "
            "this is a sign of an improperly constructed selector.");

    bool any = false;
    if (assemble_dphi_train_force_rows_d2(
            stdb, *dcs.d2b, theta_name, arg_pos,
            dphi_out, n_cols, force_row_start_per_struct))
        any = true;

    if (dcs.dmb &&
        assemble_dphi_train_force_rows_dm(
            stdb, *dcs.dmb, theta_name, arg_pos,
            dphi_out, n_cols, force_row_start_per_struct))
        any = true;

    return any;
}

/* DC_Selector-level walker. Iterates the active D2 + DM arms in
 * sequence, accumulating into the same dphi_out row. RCUT2B / RCUTMB
 * keys are descriptor-specific so a given (theta_name, arg_pos) is
 * owned by at most one arm; cross-cutting fan-out within an arm is
 * handled by the per-descriptor calc_dDdtheta / calc_daed_dtheta. */
inline bool
assemble_dphi_train_energy_rows(const StructureDB&                stdb,
                                const tadah::models::DC_Selector& dcs,
                                const std::string&                theta_name,
                                int                               arg_pos,
                                double*                           dphi_out,
                                std::size_t                       n_cols)
{
    if (!dcs.d2b)
        throw std::runtime_error(
            "assemble_dphi_train_energy_rows: DC_Selector has no D2 descriptor; "
            "DC_Selector::init always allocates a D2_Dummy fallback — "
            "this is a sign of an improperly constructed selector.");

    bool any = false;
    /* D2 arm. D2_Dummy has size 0 and its calc_dDdtheta returns
     * false (default base impl), so this is a safe no-op when
     * INIT2B=false. */
    if (assemble_dphi_train_energy_rows_d2(
            stdb, *dcs.d2b, theta_name, arg_pos, dphi_out, n_cols))
        any = true;

    /* DM arm. DM_Dummy has size 0 and its calc_drhoi_dtheta /
     * calc_daed_dtheta return false (default base impl), so this is
     * a safe no-op when INITMB=false. The walker also short-circuits
     * on dm.rhoi_size() == 0 to skip the per-atom rho scratch entirely. */
    if (dcs.dmb &&
        assemble_dphi_train_energy_rows_dm(
            stdb, *dcs.dmb, theta_name, arg_pos, dphi_out, n_cols))
        any = true;

    return any;
}

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_DPHI_DTHETA_WALK_H
