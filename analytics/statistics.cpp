#include "statistics.h"
#include <cmath>

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
    return 1- res_sum_sq(obs,pred)/tot_sum_sq(obs);
}

double Statistics::variance(const vec &v) {
    return tot_sum_sq(v)/(v.size()-1);
}

double Statistics::mean(const vec &v) {
    return v.mean();
}
