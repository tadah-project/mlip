#ifndef STATISTICS_H
#define STATISTICS_H

#include "../../CORE/core_types.h"

/** Some basis statistical tools */
class Statistics {
    using vec = aed_type2;
    public:
        /** Residual sum of squares. */
        static double res_sum_sq(const vec &obs, const vec &pred);

        /** Total sum of squares. */
        static double tot_sum_sq(const vec &obs);

        /** Coefficient of determination. */
        static double r_sq(const vec &obs, const vec &pred);

        /** Unbiased sample variance. */
        static double variance(const vec &v);

        /** Mean. */
        static double mean(const vec &v);

};
#endif
