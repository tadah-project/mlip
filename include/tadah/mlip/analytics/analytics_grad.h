#ifndef ANALYTICS_GRAD_H
#define ANALYTICS_GRAD_H

/* analytics_grad.h - per-metric closed-form derivatives for HPO gradients.
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
 *           structure), or STRESS (6 cells per structure, x<=y order: the
 *           predicted entries of the symmetric tensor)
 *   comp  = within-structure cell index (0 for ENERGY; 3*a+k for FORCE;
 *           the linearised Voigt-like 6-cell index for STRESS, matching
 *           Analytics' iteration order x in [0..3), y in [x..3))
 *   value = d(metric)/d(predicted cell), with units consistent with
 *           Analytics (eV/atom for energy metrics, eV/A for force metrics
 *           and GPa for the stress metric of the nine entries of each
 *           tensor; any post-scaling done by the host applies as a
 *           constant multiplier in the chain rule)
 *
 * Multi-segment aggregation: the mean over the measured segments,
 *
 *   scalar_avg = (1/S) * sum_{measured s} metric_s
 *
 * where metric_s is the per-segment Analytics formula (calc_e_mae /
 * calc_e_rmse / ... return a t_type of one entry per segment, nan for a
 * segment the metric does not measure) and S the number of measured
 * segments (measured_mean() of measured_datasets()): those with a
 * structure whose weight of the quantity is not 0. Each helper computes
 * d(scalar_avg)/d(cell) directly. For single-segment configs (S = 1) the
 * formulas reduce to the original byte-for-byte.
 *
 * Sign conventions match Analytics:
 *   E_diff_i      = E_truth_i - E_pred_i              (per-structure)
 *   F_diff_iak    = F_truth_iak - F_pred_iak          (per-component)
 *   S_diff_ixy    = S_truth_ixy - S_pred_ixy          (per-entry, x<=y)
 *
 * with d/dpred = -d/ddiff for every metric below. The closed-form
 * derivatives are derived directly from the Analytics formulas in
 * MLIP/src/analytics.cpp.
 */

#include <tadah/core/core_types.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/row_weights.h>  // kGPaInEvPerA3

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
/* The stress cells: the six predicted entries x <= y of the symmetric
 * tensor, of which the stress metric reads nine (srmse_gpa_per_dataset: an
 * off-diagonal cell's partial carries both of its entries). */
inline std::size_t stress_cells_per_struct(const Structure&) { return 6; }

/* The energy cells: a structure stored with energy priority 0 (eweight) has
 * no energy of its own and is left out, as in Analytics' energy metrics
 * (counts_in_metric(s, MetricQuantity::Energy)); a segment without such a
 * structure is not measured and emits nothing, and the scalar is the mean
 * over the measured segments (S below), as measured_mean() takes it. */
inline bool energy_counts(const Structure& s) {
    return counts_in_metric(s, MetricQuantity::Energy);
}

/* The energy-counting structures of segment [lo, hi). */
inline std::size_t energy_count(const StructureDB& truth, std::size_t lo,
                                std::size_t hi) {
    std::size_t n = 0;
    for (std::size_t i = lo; i < hi; ++i) n += energy_counts(truth(i));
    return n;
}

/* The segments with an energy-counting structure. */
inline std::size_t energy_measured_segments(const StructureDB& truth) {
    std::size_t S = 0;
    for (std::size_t s = 0; s + 1 < truth.dbidx.size(); ++s)
        S += energy_count(truth, truth.dbidx[s], truth.dbidx[s + 1]) > 0;
    return S;
}

/* d(emae)/d(E_pred_i) where
 *   metric_s = (1/N_s) * sum_{i in s, eweight_i != 0} |E_truth_i - E_pred_i| / natoms_i
 *   scalar   = (1/S) * sum_{measured s} metric_s
 *
 * N_s counts the structures of segment s with eweight != 0, S the
 * segments with N_s > 0. The per-structure contribution is
 * sgn(diff)/(N_s * natoms_i) divided by S. With d/dpred = -d/ddiff, the
 * cell partial is -sgn(diff)/(S * N_s * natoms_i).
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
    if (truth.size() == 0) return;
    const std::size_t S = energy_measured_segments(truth);
    for (std::size_t s = 0; s + 1 < truth.dbidx.size(); ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_s = energy_count(truth, lo, hi);
        if (N_s == 0) continue;
        for (std::size_t i = lo; i < hi; ++i) {
            if (!energy_counts(truth(i))) continue;
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
 *   metric_s = sqrt((1/N_s) * sum_{i in s, eweight_i != 0} e_i^2)
 *   scalar   = (1/S) * sum_{measured s} metric_s
 *
 * Then d(metric_s)/d(E_pred_i) = -e_i/(N_s * metric_s * natoms_i) for
 * i in segment s with eweight_i != 0, and d(scalar)/d(E_pred_i) = (1/S) *
 * d(metric_s)/... If metric_s is zero (perfect fit on that segment) the
 * gradient is formally undefined; we emit zero for that segment, matching
 * the FD limit at a flat minimum. Other segments are unaffected. */
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
    std::vector<std::size_t> N_per_seg(S, 0);
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_s = energy_count(truth, lo, hi);
        N_per_seg[s] = N_s;
        if (N_s == 0) continue;  // not measured
        double sumsq = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            if (!energy_counts(truth(i))) continue;
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
        const std::size_t N_s = N_per_seg[s];
        for (std::size_t i = lo; i < hi; ++i) {
            if (!energy_counts(truth(i))) continue;
            const double n = static_cast<double>(truth(i).natoms());
            const double e = (truth(i).energy - pred(i).energy) / n;
            const double v = -scale * e
                           / (static_cast<double>(N_s)
                              * metric_per_seg[s] * n);
            set_dydp(s, i, PredCellKind::ENERGY, std::size_t{0}, v);
        }
    }
}

// Compatibility wrapper: derivative of the arithmetic mean of the dataset
// metrics over the measured datasets (a dataset without an energy is not
// measured: it emits nothing and is not counted).
template<class F>
void ermse(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    const std::size_t count = energy_measured_segments(truth);
    ermse_per_dataset(truth, pred, scale,
        [&](std::size_t, std::size_t i, PredCellKind kind, std::size_t comp, double partial) {
            set_dydp(i, kind, comp, partial / static_cast<double>(count));
        });
}

/* d(errmse)/d(E_pred_i) where
 *   rr_i     = (E_truth_i - E_pred_i)/E_truth_i   (skip if E_truth_i==0 or eweight_i==0)
 *   metric_s = sqrt((1/N_s^eff) * sum_{i in s counted} rr_i^2)
 *   scalar   = (1/S) * sum_{measured s} metric_s
 *
 * Structures with zero truth energy or eweight 0 are skipped (they
 * contribute nothing to the metric and zero to the gradient). N_s^eff is
 * the per-segment count of the others, S the segments with N_s^eff > 0
 * (Analytics::calc_e_rrmse is nan for the rest). */
template<class F>
void errmse(const StructureDB& truth, const StructureDB& pred,
            double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::errmse: size mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S_all = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    auto counted = [&](std::size_t i) {
        return truth(i).energy != 0.0 && energy_counts(truth(i));
    };
    std::vector<double> metric_per_seg(S_all, 0.0);
    std::vector<std::size_t> N_eff_per_seg(S_all, 0);
    std::size_t S = 0;
    for (std::size_t s = 0; s < S_all; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_eff = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            if (!counted(i)) continue;
            const double rr = (truth(i).energy - pred(i).energy) / truth(i).energy;
            sumsq += rr * rr;
            ++N_eff;
        }
        N_eff_per_seg[s] = N_eff;
        if (N_eff > 0) {
            metric_per_seg[s] = std::sqrt(sumsq / static_cast<double>(N_eff));
            ++S;
        }
    }
    for (std::size_t s = 0; s < S_all; ++s) {
        if (!(metric_per_seg[s] > 0.0)) continue;
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_eff = N_eff_per_seg[s];
        for (std::size_t i = lo; i < hi; ++i) {
            if (!counted(i)) continue;
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
 *   SS_res_s   = sum_{i in s, eweight_i != 0} (obs_i - pred_i)^2
 *   SS_tot_s   = sum_{i in s, eweight_i != 0} (obs_i - mean_obs_s)^2,
 *                mean_obs_s over the same structures
 *   metric_s   = 1 - SS_res_s / SS_tot_s
 *   scalar     = (1/S) * sum_{measured s} metric_s
 *
 * Only pred_i depends on E_pred_i, so:
 *   d(SS_res_s)/d(E_pred_i) = -2 * (obs_i - pred_i) / natoms_i
 *   d(metric_s)/d(E_pred_i) = (2 / SS_tot_s) * (obs_i - pred_i) / natoms_i
 *   d(scalar)/d(E_pred_i)   = 1/S * d(metric_s)/d(E_pred_i)
 *
 * S counts the segments with a structure of eweight != 0. If SS_tot_s ==
 * 0 (constant truth in segment s), R^2 is degenerate on that segment;
 * emit zero gradient there. Other segments unaffected. */
template<class F>
void ersq(const StructureDB& truth, const StructureDB& pred,
          double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::ersq: size mismatch");
    if (truth.dbidx.size() < 2) return;
    if (truth.size() == 0) return;
    const std::size_t S = energy_measured_segments(truth);
    for (std::size_t s = 0; s + 1 < truth.dbidx.size(); ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_s = energy_count(truth, lo, hi);
        if (N_s == 0) continue;
        double mean_obs = 0.0;
        for (std::size_t i = lo; i < hi; ++i)
            if (energy_counts(truth(i)))
                mean_obs += truth(i).energy / static_cast<double>(truth(i).natoms());
        mean_obs /= static_cast<double>(N_s);
        double SS_tot = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            if (!energy_counts(truth(i))) continue;
            const double obs = truth(i).energy
                             / static_cast<double>(truth(i).natoms());
            SS_tot += (obs - mean_obs) * (obs - mean_obs);
        }
        if (!(SS_tot > 0.0)) continue;
        for (std::size_t i = lo; i < hi; ++i) {
            if (!energy_counts(truth(i))) continue;
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
 *   metric_s = sqrt((1/N_s^F) * sum_{i in s with fweight != 0, a, k} (F_truth_iak - F_pred_iak)^2)
 *   scalar   = (1/S) * sum_s metric_s
 *
 * N_s^F = sum_{i in s with fweight != 0} 3 * natoms_i is the per-segment
 * force-cell count. A structure with force priority 0 (counts_in_metric())
 * is left out, as in Analytics::calc_f_rmse: none of its cells is emitted;
 * a segment without such structures (N_s^F = 0) is not measured and emits
 * nothing.
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
    require_predicted_forces(truth, pred, "the gradient of the force RMSE");
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
            if (!counts_in_metric(truth(i), MetricQuantity::Forces)) continue;
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
            if (!counts_in_metric(truth(i), MetricQuantity::Forces)) continue;
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

// Compatibility wrapper: derivative of the arithmetic mean of the dataset
// metrics over the measured datasets (a dataset without forces is not
// measured: it emits nothing and is not counted).
template<class F>
void frmse(const StructureDB& truth, const StructureDB& pred,
           double scale, F set_dydp)
{
    std::size_t count = 0;
    for (bool m : measured_datasets(truth, MetricQuantity::Forces)) count += m;
    frmse_per_dataset(truth, pred, scale,
        [&](std::size_t, std::size_t i, PredCellKind kind, std::size_t comp, double partial) {
            set_dydp(i, kind, comp, partial / static_cast<double>(count));
        });
}

/* d(srmse_gpa_s)/d(S_pred_ixy) for each dataset s, where
 *   metric_s = c * sqrt((1/(9 n_s)) * sum_{i in s with sweight != 0}
 *                       sum_{x <= y} m_xy (S_truth_ixy - S_pred_ixy)^2)
 * is Analytics::calc_s_rmse_gpa: the root mean square of the nine entries
 * of each residual tensor (m_xy = 2 for an off-diagonal entry, which the
 * symmetric tensor has twice, 1 on the diagonal), in GPa (c =
 * 1/kGPaInEvPerA3: the data hold eV/A^3), n_s the structures of s with a
 * stress tensor. A cell is one of the six predicted entries x <= y
 * (stress_cells_per_struct; an off-diagonal entry of the symmetric
 * prediction is one value), its partial
 *   -scale * c^2 * m_xy * (S_truth_ixy - S_pred_ixy) / (9 n_s metric_s).
 * A structure with stress priority 0 (counts_in_metric()) is left out, as
 * in Analytics: none of its cells is emitted; a dataset without a stress
 * tensor is not measured and emits nothing. set_dydp(s, i, kind, comp,
 * partial) as frmse_per_dataset's: no average over the datasets, which a
 * PC_SRMSE_GPA line with dataset targets weights one by one. */
template<class F>
void srmse_gpa_per_dataset(const StructureDB& truth, const StructureDB& pred,
                           double scale, F set_dydp)
{
    if (truth.size() != pred.size())
        throw std::runtime_error("analytics_grad::srmse_gpa: size mismatch");
    if (truth.dbidx != pred.dbidx)
        throw std::runtime_error("analytics_grad::srmse_gpa: dataset layout mismatch");
    if (truth.dbidx.size() < 2) return;
    const std::size_t S = truth.dbidx.size() - 1;
    if (truth.size() == 0) return;
    const double c = 1.0 / kGPaInEvPerA3;
    auto m_of = [](std::size_t x, std::size_t y) { return x == y ? 1.0 : 2.0; };
    std::vector<double> metric_per_seg(S, 0.0);
    std::vector<std::size_t> n_per_seg(S, 0);
    for (std::size_t s = 0; s < S; ++s) {
        double sumsq = 0.0;
        std::size_t n = 0;
        for (std::size_t i = truth.dbidx[s]; i < truth.dbidx[s + 1]; ++i) {
            if (!counts_in_metric(truth(i), MetricQuantity::Stress)) continue;
            ++n;
            for (std::size_t x = 0; x < 3; ++x)
                for (std::size_t y = x; y < 3; ++y) {
                    const double d = truth(i).stress(x, y) - pred(i).stress(x, y);
                    sumsq += m_of(x, y) * d * d;
                }
        }
        n_per_seg[s] = n;
        if (n > 0)
            metric_per_seg[s] = c * std::sqrt(sumsq / (9.0 * static_cast<double>(n)));
    }
    for (std::size_t s = 0; s < S; ++s) {
        if (!(metric_per_seg[s] > 0.0)) continue;
        const double n = static_cast<double>(n_per_seg[s]);
        for (std::size_t i = truth.dbidx[s]; i < truth.dbidx[s + 1]; ++i) {
            if (!counts_in_metric(truth(i), MetricQuantity::Stress)) continue;
            std::size_t comp = 0;
            for (std::size_t x = 0; x < 3; ++x)
                for (std::size_t y = x; y < 3; ++y, ++comp) {
                    const double d = truth(i).stress(x, y) - pred(i).stress(x, y);
                    set_dydp(s, i, PredCellKind::STRESS, comp,
                             -scale * c * c * m_of(x, y) * d
                                 / (9.0 * n * metric_per_seg[s]));
                }
        }
    }
}

/* d(srmse_gpa)/d(S_pred_ixy) of the mean of the per-dataset metrics of
 * srmse_gpa_per_dataset over the measured datasets (those with a stress
 * tensor), as Analytics' measured_mean() takes it: each partial divided by
 * their number. */
template<class F>
void srmse_gpa(const StructureDB& truth, const StructureDB& pred,
               double scale, F set_dydp)
{
    std::size_t count = 0;
    for (bool m : measured_datasets(truth, MetricQuantity::Stress)) count += m;
    srmse_gpa_per_dataset(truth, pred, scale,
        [&](std::size_t, std::size_t i, PredCellKind kind, std::size_t comp, double partial) {
            set_dydp(i, kind, comp, partial / static_cast<double>(count));
        });
}

}  // namespace analytics_grad

}  // namespace mlip
}  // namespace tadah

#endif  // ANALYTICS_GRAD_H
