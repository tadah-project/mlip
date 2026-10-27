#ifndef STATISTICS_H
#define STATISTICS_H

#include <tadah/core/core_types.h>
namespace tadah {
namespace mlip {

/** Some basis statistical tools */
class Statistics {
    using vec = tadah::core::aed_type;
    public:
        /** Residual sum of squares. */
        static double res_sum_sq(const vec &obs, const vec &pred);

        /** Total sum of squares. */
        static double tot_sum_sq(const vec &obs);

        /** Coefficient of determination, 1 - SS_res / SS_tot; NaN when
         *  the observations are all equal (or fewer than two): they have
         *  no spread to measure the residual against. */
        static double r_sq(const vec &obs, const vec &pred);

        /** Unbiased sample variance. */
        static double variance(const vec &v);

        /** Mean. */
        static double mean(const vec &v);

};
} // namespace mlip
} // namespace tadah
#endif
