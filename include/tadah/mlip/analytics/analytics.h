#ifndef ANALYTICS_H
#define ANALYTICS_H

#include <tadah/mlip/structure_db.h>
#include <tadah/core/core_types.h>

#include <string>
#include <vector>
namespace tadah {
namespace mlip {

/** Refuses a prediction that lacks a force for some reference atom.
 *
 *  The force metrics compare truth(i).atoms[a].force with
 *  pred(i).atoms[a].force for every reference atom, so pred(i) must hold
 *  exactly truth(i).natoms() atoms. A prediction made without forces
 *  (M_Tadah_Base::predict with FORCE false leaves Structure::atoms empty)
 *  or with another number of atoms throws std::runtime_error naming
 *  @p who, the structure (1-based, "structure 2 of 3"), its label and both
 *  atom counts. The containers must already have the same size.
 */
void require_predicted_forces(const StructureDB &truth,
                              const StructureDB &pred,
                              const char *who);

/** What a metric measures: the energies, the forces or the stresses. */
enum class MetricQuantity { Forces, Stress, Energy };

/** Whether structure @p s counts in the @p q metrics: a structure stored
 *  with energy weight 0 (eweight) has no energy of its own, one with force
 *  weight 0 (fweight) no force data and one with stress weight 0 (sweight)
 *  no stress tensor (tadah data convert stores placeholders for them); the
 *  metrics leave them out, as training does. */
inline bool counts_in_metric(const Structure &s, MetricQuantity q) {
    const double w = q == MetricQuantity::Energy   ? s.eweight
                     : q == MetricQuantity::Forces ? s.fweight
                                                   : s.sweight;
    return w != 0.0;
}

/** For each dataset of @p truth, whether the @p q metrics measure it: a
 *  dataset in which no structure counts (every one has that weight 0, or
 *  none is left) is not measured; its metric is nan. */
std::vector<bool> measured_datasets(const StructureDB &truth, MetricQuantity q);

/** The mean of the per-dataset values @p v over the datasets @p measured
 *  marks (with every dataset marked, v.mean() itself); nan when it marks
 *  none. */
double measured_mean(const tadah::core::t_type &v, const std::vector<bool> &measured);

/** Refuses @p truth when no dataset is measured by the @p q metrics (every
 *  structure has that weight 0): they would have nothing to measure. Throws
 *  std::runtime_error, prefixed with @p who, naming the files when
 *  @p dataset_names has them and the weight ("energy weight 0 (eweight)",
 *  "force weight 0 (fweight)", "stress weight 0 (sweight)"). A dataset of
 *  only such structures among others is not refused: it is not measured
 *  (measured_datasets()). */
void require_metric_structures(const StructureDB &truth, MetricQuantity q,
                               const std::string &who,
                               const std::vector<std::string> &dataset_names = {});

/** Class for analysing and comparing datasets
 *
 *  Each metric is per dataset (DBFILE). The energy metrics (calc_e_*) leave
 *  out the structures stored with energy weight 0, the force metrics
 *  (calc_f_*) those with force weight 0, and the stress metrics (calc_s_*)
 *  those with stress weight 0 (counts_in_metric()): they have no data of
 *  their own. A dataset without a structure that counts is not measured:
 *  its value is nan (measured_datasets(), measured_mean()). calc_e_rrmse
 *  also leaves out a structure whose reference energy is 0.
 */
class Analytics {
    private:
        const StructureDB &st;    // orig data
        const StructureDB &stp;   // predicted data

    public:
        Analytics(const StructureDB &st, const StructureDB &stp);

        /** Return Energy/atom Mean Absolute Error for each DBFILE. */
        tadah::core::t_type calc_e_mae() const;

        /** Return Force Mean Absolute Error for each DBFILE.
         *
         *  The force metrics need a predicted force for every reference
         *  atom (require_predicted_forces()). */
        tadah::core::t_type calc_f_mae() const;

        /** Return Stress Mean Absolute Error for each DBFILE.
         *
         *  Calculated using 6 independent components.
         */
        tadah::core::t_type calc_s_mae() const;

        /** Return Energy/atom Relative Root Mean Square Error for each DBFILE. */
        tadah::core::t_type calc_e_rrmse() const;

        /** Return Energy/atom Root Mean Square Error for each DBFILE. */
        tadah::core::t_type calc_e_rmse() const;

        /** Return Force Root Mean Square Error for each DBFILE. */
        tadah::core::t_type calc_f_rmse() const;

        /** Return Stress Root Mean Square Error for each DBFILE.
         *
         *  Calculated using 6 independent components.
         */
        tadah::core::t_type calc_s_rmse() const;

        /** Return Energy/atom coefficient of determination for each DBFILE. */
        tadah::core::t_type calc_e_r_sq() const;

        /** Return Force coefficient of determination for each DBFILE. */
        tadah::core::t_type calc_f_r_sq() const;

        /** Return Stress coefficient of determination for each DBFILE. */
        tadah::core::t_type calc_s_r_sq() const;

};
} // namespace mlip
} // namespace tadah
#endif
