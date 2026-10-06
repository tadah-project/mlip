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

/** What a force or stress metric measures. */
enum class MetricQuantity { Forces, Stress };

/** Whether structure @p s counts in the @p q metrics: a structure stored
 *  with force weight 0 (fweight) has no force data of its own, and one with
 *  stress weight 0 (sweight) no stress tensor (tadah data convert stores
 *  them so); the metrics leave them out, as training does. */
inline bool counts_in_metric(const Structure &s, MetricQuantity q) {
    return (q == MetricQuantity::Forces ? s.fweight : s.sweight) != 0.0;
}

/** Refuses a dataset of @p truth in which no structure counts in the @p q
 *  metrics (every one has that weight 0): its metric would have nothing to
 *  measure. Throws std::runtime_error, prefixed with @p who, naming the
 *  first such dataset ("dataset 2 of 3", and its file when
 *  @p dataset_names has it) and the weight ("force weight 0 (fweight)",
 *  "stress weight 0 (sweight)"). */
void require_metric_structures(const StructureDB &truth, MetricQuantity q,
                               const std::string &who,
                               const std::vector<std::string> &dataset_names = {});

/** Class for analysing and comparing datasets
 *
 *  Each metric is per dataset (DBFILE). The force metrics (calc_f_*) leave
 *  out the structures stored with force weight 0, and the stress metrics
 *  (calc_s_*) those with stress weight 0 (counts_in_metric()): they have no
 *  data of their own. A dataset without a structure that counts is refused
 *  (require_metric_structures()). The energy metrics count every structure.
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
