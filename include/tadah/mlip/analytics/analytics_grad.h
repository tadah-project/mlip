#ifndef ANALYTICS_GRAD_H
#define ANALYTICS_GRAD_H

/* analytics_grad.h - per-metric closed-form derivatives for HPO Phase 2.
 *
 * Each helper takes the (truth, predicted) StructureDB pair the same way
 * Analytics does, plus a callback `set_dydp` that the helper invokes
 * once for every (i, kind, comp) cell of the predicted side, with the
 * partial derivative of the *named metric* with respect to that cell.
 *
 *   set_dydp(i, kind, comp, value)
 *
 * where:
 *   i     = structure index in the validation StructureDB
 *   kind  = ENERGY (one cell per structure), FORCE (3 * natoms cells per
 *           structure), or STRESS (6 cells per structure, x<=y order)
 *   comp  = within-structure cell index (0 for ENERGY; 3*a+k for FORCE;
 *           the linearised Voigt-like 6-cell index for STRESS, matching
 *           Analytics' iteration order x in [0..3), y in [x..3))
 *   value = d(metric)/d(predicted cell), with units consistent with
 *           Analytics (eV/atom for energy metrics, raw eV/A and GPa for
 *           force / stress unless the host scales the metric afterward;
 *           any post-scaling done by the host applies as a constant
 *           multiplier in the chain rule)
 *
 * Multi-segment aggregation mirrors HPO_Host::fill_validation:
 *
 *   scalar_avg = (1/S) * sum_{s=0..S-1} metric_s
 *
 * where S = truth.dbidx.size() - 1 and metric_s is the per-segment
 * Analytics formula (calc_e_mae / calc_e_rmse / ... return a t_type of
 * length S, one entry per segment, and HPO_Host averages them). Each
 * helper computes d(scalar_avg)/d(cell) directly. For single-segment
 * configs (S = 1) the formulas reduce to the original byte-for-byte.
 *
 * Sign conventions match Analytics:
 *   E_diff_i      = E_truth_i - E_pred_i              (per-structure)
 *   F_diff_iak    = F_truth_iak - F_pred_iak          (per-component)
 *   S_diff_ixy    = S_truth_ixy - S_pred_ixy          (per-component, x<=y)
 *
 * with d/dpred = -d/ddiff for every metric below. The closed-form
 * derivatives are derived directly from the Analytics formulas in
 * MLIP/src/analytics.cpp.
 */

#include <tadah/core/core_types.h>
#include <tadah/mlip/structure_db.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace tadah {
namespace mlip {

enum class PredCellKind { ENERGY, FORCE, STRESS };

namespace analytics_grad {

/* Common helper: consume the (E_pred_i, F_pred_iak, S_pred_ixy) cells
 * of a predicted StructureDB in the same order Analytics iterates. */
inline std::size_t energy_cells_per_struct(const Structure&) { return 1; }
inline std::size_t force_cells_per_struct (const Structure& s) { return 3 * s.natoms(); }
inline std::size_t stress_cells_per_struct(const Structure&) { return 6; }

/* d(emae)/d(E_pred_i) where
 *   metric_s = (1/N_s) * sum_{i in s} |E_truth_i - E_pred_i| / natoms_i
 *   scalar   = (1/S) * sum_s metric_s
 *
 * The per-structure contribution is sgn(diff)/(N_s * natoms_i) divided
 * by S (the segment count). With d/dpred = -d/ddiff, the cell partial
 * is -sgn(diff)/(S * N_s * natoms_i).
 *
 * Caller convention: pass the same `scale` the host applies to the
 * metric (e.g. 1000 for emae meV/atom). The chain rule multiplies by
 * the scale linearly. */
template<class F>
void emae(const StructureDB& truth, const StructureDB& pred,
          double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::emae: size mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        if (hi <= lo) continue;
        const std::size_t N_s = hi - lo;
        for (std::size_t i = lo; i < hi; ++i) {
            const double diff = truth(i).energy - pred(i).energy;
            const double sgn  = (diff > 0) ? 1.0 : (diff < 0) ? -1.0 : 0.0;
            const double n    = static_cast<double>(truth(i).natoms());
            const double v    = -scale * sgn
                              / (static_cast<double>(S)
                                 * static_cast<double>(N_s) * n);
            set_dydp(i, PredCellKind::ENERGY, std::size_t{0}, v);
        }
    }
}

/* d(ermse)/d(E_pred_i) where
 *   e_i      = (E_truth_i - E_pred_i)/natoms_i
 *   metric_s = sqrt((1/N_s) * sum_{i in s} e_i^2)
 *   scalar   = (1/S) * sum_s metric_s
 *
 * Then d(metric_s)/d(E_pred_i) = -e_i/(N_s * metric_s * natoms_i) for
 * i in segment s, and d(scalar)/d(E_pred_i) = (1/S) * d(metric_s)/...
 * If metric_s is zero (perfect fit on that segment) the gradient is
 * formally undefined; we emit zero for that segment, matching the FD
 * limit at a flat minimum. Other segments are unaffected. */
// Emits (dataset_index, structure_index, kind, component, partial), with
// zero-based dataset indices and NO dataset averaging in partial.
template<class F>
void ermse_per_dataset(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::ermse: size mismatch");
    if (truth.dbidx != pred.dbidx)
        throw std::runtime_error("analytics_grad::ermse: dataset layout mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    std::vector<double> metric_per_seg(S, 0.0);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        if (hi <= lo) continue;
        const std::size_t N_s = hi - lo;
        double sumsq = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double e = (truth(i).energy - pred(i).energy)
                           / static_cast<double>(truth(i).natoms());
            sumsq += e * e;
        }
        metric_per_seg[s] = std::sqrt(sumsq / static_cast<double>(N_s));
    }
    for (std::size_t s = 0; s < S; ++s) {
        if (!(metric_per_seg[s] > 0.0)) continue;
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_s = hi - lo;
        for (std::size_t i = lo; i < hi; ++i) {
            const double n = static_cast<double>(truth(i).natoms());
            const double e = (truth(i).energy - pred(i).energy) / n;
            const double v = -scale * e
                           / (static_cast<double>(N_s)
                              * metric_per_seg[s] * n);
            set_dydp(s, i, PredCellKind::ENERGY, std::size_t{0}, v);
        }
    }
}

// Compatibility wrapper: derivative of the arithmetic mean of dataset metrics.
template<class F>
void ermse(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    const auto count = truth.dbidx.size() > 1 ? truth.dbidx.size() - 1 : 0;
    ermse_per_dataset(truth, pred, scale,
        [&](std::size_t, std::size_t i, PredCellKind kind, std::size_t comp, double partial) {
            set_dydp(i, kind, comp, partial / static_cast<double>(count));
        });
}

/* d(errmse)/d(E_pred_i) where
 *   rr_i     = (E_truth_i - E_pred_i)/E_truth_i        (skip if E_truth_i==0)
 *   metric_s = sqrt((1/N_s^eff) * sum_{i in s, E_truth_i!=0} rr_i^2)
 *   scalar   = (1/S) * sum_s metric_s
 *
 * Zero-truth structures are skipped (they contribute nothing to the
 * metric and zero to the gradient). N_s^eff is the per-segment count of
 * structures with nonzero truth energy. */
template<class F>
void errmse(const StructureDB& truth, const StructureDB& pred,
            double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::errmse: size mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    std::vector<double> metric_per_seg(S, 0.0);
    std::vector<std::size_t> N_eff_per_seg(S, 0);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_eff = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            if (truth(i).energy == 0.0) continue;
            const double rr = (truth(i).energy - pred(i).energy) / truth(i).energy;
            sumsq += rr * rr;
            ++N_eff;
        }
        N_eff_per_seg[s] = N_eff;
        if (N_eff > 0)
            metric_per_seg[s] = std::sqrt(sumsq / static_cast<double>(N_eff));
    }
    for (std::size_t s = 0; s < S; ++s) {
        if (!(metric_per_seg[s] > 0.0)) continue;
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_eff = N_eff_per_seg[s];
        for (std::size_t i = lo; i < hi; ++i) {
            if (truth(i).energy == 0.0) continue;
            const double rr = (truth(i).energy - pred(i).energy) / truth(i).energy;
            const double v = -scale * rr
                           / (static_cast<double>(S)
                              * static_cast<double>(N_eff)
                              * metric_per_seg[s] * truth(i).energy);
            set_dydp(i, PredCellKind::ENERGY, std::size_t{0}, v);
        }
    }
}

/* d(ersq)/d(E_pred_i) for the coefficient-of-determination metric
 *
 *   obs_i      = E_truth_i / natoms_i,  pred_i = E_pred_i / natoms_i
 *   SS_res_s   = sum_{i in s} (obs_i - pred_i)^2
 *   SS_tot_s   = sum_{i in s} (obs_i - mean_obs_s)^2,  mean_obs_s per-segment
 *   metric_s   = 1 - SS_res_s / SS_tot_s
 *   scalar     = (1/S) * sum_s metric_s
 *
 * Only pred_i depends on E_pred_i, so:
 *   d(SS_res_s)/d(E_pred_i) = -2 * (obs_i - pred_i) / natoms_i
 *   d(metric_s)/d(E_pred_i) = (2 / SS_tot_s) * (obs_i - pred_i) / natoms_i
 *   d(scalar)/d(E_pred_i)   = 1/S * d(metric_s)/d(E_pred_i)
 *
 * If SS_tot_s == 0 (constant truth in segment s), R^2 is degenerate on
 * that segment; emit zero gradient there. Other segments unaffected. */
template<class F>
void ersq(const StructureDB& truth, const StructureDB& pred,
          double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::ersq: size mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        if (hi <= lo) continue;
        const std::size_t N_s = hi - lo;
        double mean_obs = 0.0;
        for (std::size_t i = lo; i < hi; ++i)
            mean_obs += truth(i).energy / static_cast<double>(truth(i).natoms());
        mean_obs /= static_cast<double>(N_s);
        double SS_tot = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double obs = truth(i).energy
                             / static_cast<double>(truth(i).natoms());
            SS_tot += (obs - mean_obs) * (obs - mean_obs);
        }
        if (!(SS_tot > 0.0)) continue;
        for (std::size_t i = lo; i < hi; ++i) {
            const double n   = static_cast<double>(truth(i).natoms());
            const double obs = truth(i).energy / n;
            const double prd = pred (i).energy / n;
            const double v   = scale * 2.0 * (obs - prd)
                             / (static_cast<double>(S) * SS_tot * n);
            set_dydp(i, PredCellKind::ENERGY, std::size_t{0}, v);
        }
    }
}

/* d(frmse)/d(F_pred_iak) where
 *   metric_s = sqrt((1/N_s^F) * sum_{i in s, a, k} (F_truth_iak - F_pred_iak)^2)
 *   scalar   = (1/S) * sum_s metric_s
 *
 * N_s^F = sum_{i in s} 3 * natoms_i is the per-segment force-cell count.
 * Per-segment partitioning aligns with the energy partition because
 * Analytics::calc_f_rmse uses truth.dbidx to detect segment boundaries:
 * each segment's frmse is built from the force cells of its structures
 * only. */
// Emits (dataset_index, structure_index, kind, component, partial), with
// zero-based dataset indices and NO dataset averaging in partial.
template<class F>
void frmse_per_dataset(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::frmse: size mismatch");
    if (truth.dbidx != pred.dbidx)
        throw std::runtime_error("analytics_grad::frmse: dataset layout mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    std::vector<double> metric_per_seg(S, 0.0);
    std::vector<std::size_t> N_F_per_seg(S, 0);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_F = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            for (std::size_t a = 0; a < truth(i).natoms(); ++a) {
                for (std::size_t k = 0; k < 3; ++k) {
                    const double d = truth(i).atoms[a].force(k)
                                   - pred(i).atoms[a].force(k);
                    sumsq += d * d;
                    ++N_F;
                }
            }
        }
        N_F_per_seg[s] = N_F;
        if (N_F > 0)
            metric_per_seg[s] = std::sqrt(sumsq / static_cast<double>(N_F));
    }
    for (std::size_t s = 0; s < S; ++s) {
        if (!(metric_per_seg[s] > 0.0)) continue;
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_F = N_F_per_seg[s];
        for (std::size_t i = lo; i < hi; ++i) {
            for (std::size_t a = 0; a < truth(i).natoms(); ++a) {
                for (std::size_t k = 0; k < 3; ++k) {
                    const double d = truth(i).atoms[a].force(k)
                                   - pred(i).atoms[a].force(k);
                    const double v = -scale * d
                                   / (static_cast<double>(N_F)
                                      * metric_per_seg[s]);
                    const std::size_t comp = 3 * a + k;
                    set_dydp(s, i, PredCellKind::FORCE, comp, v);
                }
            }
        }
    }
}

// Compatibility wrapper: derivative of the arithmetic mean of dataset metrics.
template<class F>
void frmse(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    const auto count = truth.dbidx.size() > 1 ? truth.dbidx.size() - 1 : 0;
    frmse_per_dataset(truth, pred, scale,
        [&](std::size_t, std::size_t i, PredCellKind kind, std::size_t comp, double partial) {
            set_dydp(i, kind, comp, partial / static_cast<double>(count));
        });
}

/* d(srmse)/d(S_pred_ixy) where
 *   metric_s = sqrt((1/N_s^S) * sum_{i in s, x, y>=x} (S_truth_ixy - S_pred_ixy)^2)
 *   scalar   = (1/S_seg) * sum_s metric_s
 *
 * N_s^S = 6 * (#structures in segment s) is the per-segment stress-cell
 * count. Voigt-like 6-cell ordering matches Analytics:
 * (xx, xy, xz, yy, yz, zz). */
template<class F>
void srmse(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::srmse: size mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    std::vector<double> metric_per_seg(S, 0.0);
    std::vector<std::size_t> N_S_per_seg(S, 0);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_S_cells = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            for (std::size_t x = 0; x < 3; ++x) {
                for (std::size_t y = x; y < 3; ++y) {
                    const double d = truth(i).stress(x, y) - pred(i).stress(x, y);
                    sumsq += d * d;
                    ++N_S_cells;
                }
            }
        }
        N_S_per_seg[s] = N_S_cells;
        if (N_S_cells > 0)
            metric_per_seg[s] = std::sqrt(sumsq / static_cast<double>(N_S_cells));
    }
    for (std::size_t s = 0; s < S; ++s) {
        if (!(metric_per_seg[s] > 0.0)) continue;
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_S_cells = N_S_per_seg[s];
        for (std::size_t i = lo; i < hi; ++i) {
            std::size_t comp = 0;
            for (std::size_t x = 0; x < 3; ++x) {
                for (std::size_t y = x; y < 3; ++y) {
                    const double d = truth(i).stress(x, y) - pred(i).stress(x, y);
                    const double v = -scale * d
                                   / (static_cast<double>(S)
                                      * static_cast<double>(N_S_cells)
                                      * metric_per_seg[s]);
                    set_dydp(i, PredCellKind::STRESS, comp, v);
                    ++comp;
                }
            }
        }
    }
}

}  // namespace analytics_grad

}  // namespace mlip
}  // namespace tadah

#endif  // ANALYTICS_GRAD_H
