#ifndef ANALYTICS_H
#define ANALYTICS_H

#include "../structure_db.h"
#include "../../CORE/core_types.h"


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
        t_type calc_e_mae() const;

        /** Return Force Mean Absolute Error for each DBFILE. */
        t_type calc_f_mae() const;

        /** Return Stress Mean Absolute Error for each DBFILE.
         *
         *  Calculated using 6 independed components.
         */
        t_type calc_s_mae() const;

        /** Return Energy/atom Root Mean Square Error for each DBFILE. */
        t_type calc_e_rmse() const;

        /** Return Force Root Mean Square Error for each DBFILE. */
        t_type calc_f_rmse() const;

        /** Return Stress Root Mean Square Error for each DBFILE.
         *
         *  Calculated using 6 independed components.
         */
        t_type calc_s_rmse() const;

        /** Return Energy/atom coefficient of determination for each DBFILE. */
        t_type calc_e_r_sq() const;

        /** Return Force coefficient of determination for each DBFILE. */
        t_type calc_f_r_sq() const;

        /** Return Stress coefficient of determination for each DBFILE. */
        t_type calc_s_r_sq() const;

};
#endif
