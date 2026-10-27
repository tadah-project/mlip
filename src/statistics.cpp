#include <tadah/mlip/analytics/statistics.h>

#include <cmath>
#include <limits>
namespace tadah {
namespace mlip {

double Statistics::res_sum_sq(const vec &obs, const vec &pred) {
    if (obs.size() != pred.size())
        throw std::runtime_error("Containers differ in size.");

    double rss=0;
    for (size_t i=0; i<obs.size(); ++i) {
        rss += std::pow(obs[i]-pred[i],2);
    }
    return rss;
}

double Statistics::tot_sum_sq(const vec &obs) {
    double m = obs.mean();
    double tss=0;
    //for (auto v : obs) tss+=pow(v-m,2);
    for (size_t i=0; i<obs.size(); ++i) tss+=pow(obs(i)-m,2);
    return tss;

}

double Statistics::r_sq(const vec &obs, const vec &pred) {
    /* R^2 measures the residual against the spread of the observations
     * about their mean. Observations that are all equal (forces of ideal
     * crystals, one structure's energy) have none, and R^2 has no value:
     * NaN, not 1 - SS_res / 0 (-inf, or NaN for 0/0), nor, when the mean
     * of the equal values is rounded (0.1 three times), 1 - SS_res over a
     * few ulp. Equality is tested on the values themselves. */
    bool spread = false;
    for (size_t i = 1; i < obs.size() && !spread; ++i)
        spread = obs[i] != obs[0];
    if (!spread)
        return std::numeric_limits<double>::quiet_NaN();
    return 1- res_sum_sq(obs,pred)/tot_sum_sq(obs);
}

double Statistics::variance(const vec &v) {
    return tot_sum_sq(v)/(v.size()-1);
}

double Statistics::mean(const vec &v) {
    return v.mean();
}
} // namespace mlip
} // namespace tadah
