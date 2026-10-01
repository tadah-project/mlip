#ifndef ANALYTICS_H
#define ANALYTICS_H

#include <tadah/mlip/structure_db.h>
#include <tadah/core/core_types.h>
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

/** Class for analysing and comparing datasets
 *
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
