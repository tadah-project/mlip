#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/statistics.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
namespace tadah {
namespace mlip {

void require_predicted_forces(const StructureDB &truth,
                              const StructureDB &pred,
                              const char *who)
{
    for (size_t i=0; i<truth.size(); ++i) {
        const size_t n_ref = truth(i).natoms();
        const size_t n_pred = pred(i).atoms.size();
        if (n_pred == n_ref) continue;
        std::string msg = std::string(who) + ": structure "
            + std::to_string(i+1) + " of " + std::to_string(truth.size());
        if (!truth(i).label.empty())
            msg += " ('" + truth(i).label + "')";
        msg += " has " + std::to_string(n_pred) + " predicted forces for its "
            + std::to_string(n_ref) + " atoms; a force metric needs a "
              "prediction made with forces for every atom";
        throw std::runtime_error(msg);
    }
}

std::vector<bool> measured_datasets(const StructureDB &truth, MetricQuantity q)
{
    const std::size_t n = truth.dbidx.size() > 1 ? truth.dbidx.size() - 1 : 0;
    std::vector<bool> out(n, false);
    for (std::size_t d = 0; d < n; ++d)
        for (std::size_t i = truth.dbidx[d]; i < truth.dbidx[d + 1] && !out[d]; ++i)
            out[d] = counts_in_metric(truth(i), q);
    return out;
}

double measured_mean(const tadah::core::t_type &v, const std::vector<bool> &measured)
{
    if (measured.size() != v.size())
        throw std::logic_error("measured_mean: " + std::to_string(v.size()) + " values, "
                               + std::to_string(measured.size()) + " datasets");
    if (std::find(measured.begin(), measured.end(), false) == measured.end())
        return v.mean();  // every dataset: the plain mean, bit for bit
    double sum = 0.0;
    std::size_t n = 0;
    for (std::size_t d = 0; d < v.size(); ++d)
        if (measured[d]) { sum += v[d]; ++n; }
    return n ? sum / static_cast<double>(n) : std::numeric_limits<double>::quiet_NaN();
}

void require_metric_structures(const StructureDB &truth, MetricQuantity q,
                               const std::string &who,
                               const std::vector<std::string> &dataset_names)
{
    for (bool m : measured_datasets(truth, q))
        if (m) return;
    const char *data = q == MetricQuantity::Energy   ? "a structure with an energy"
                       : q == MetricQuantity::Forces ? "a structure with forces"
                                                     : "a stress tensor";
    const char *weight = q == MetricQuantity::Energy   ? "energy priority 0 (eweight)"
                         : q == MetricQuantity::Forces ? "force priority 0 (fweight)"
                                                       : "stress priority 0 (sweight)";
    const char *metrics = q == MetricQuantity::Energy   ? "energy"
                          : q == MetricQuantity::Forces ? "force"
                                                        : "stress";
    std::string msg = who + ": no dataset has " + data;
    if (!dataset_names.empty()) {
        msg += " (";
        for (std::size_t d = 0; d < dataset_names.size(); ++d)
            msg += (d ? ", " : "") + dataset_names[d];
        msg += ")";
    }
    msg += std::string(": every structure has ") + weight + ", so the " + metrics
        + " metrics have nothing to measure";
    throw std::runtime_error(msg);
}

Analytics::Analytics(const StructureDB &st, const StructureDB &stp):
    st(st),
    stp(stp)
{
    if (st.size()!=stp.size())
        throw std::runtime_error("Containers differ in size.");
}

tadah::core::t_type Analytics::calc_e_mae() const{

    tadah::core::t_type emae_vec(st.dbidx.size()-1);
    double emae=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        // assume stp(i).natoms()==st(i).natoms()
        if (counts_in_metric(st(i), MetricQuantity::Energy)) {
            emae += std::abs(st(i).energy - stp(i).energy)/st(i).natoms();
            N++;
        }
        if (i+1==st.dbidx[dbidx+1]) {
            emae_vec(dbidx)=N ? emae/N : std::numeric_limits<double>::quiet_NaN();  // not measured
            emae=0;
            N=0;
            dbidx++;
        }
    }
    return emae_vec;
}

tadah::core::t_type Analytics::calc_f_mae() const{
    require_predicted_forces(st, stp, "Analytics::calc_f_mae");

    tadah::core::t_type fmae_vec(st.dbidx.size()-1);
    double fmae=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        const bool counts = counts_in_metric(st(i), MetricQuantity::Forces);
        for (size_t a=0; counts && a<st(i).natoms(); ++a) {
            for (size_t k=0; k<3; ++k) {
                fmae += std::abs(st(i).atoms[a].force(k) - stp(i).atoms[a].force(k));
                N++;
            }
        }

        if (i+1==st.dbidx[dbidx+1]) {
            fmae_vec(dbidx)=N ? fmae/N : std::numeric_limits<double>::quiet_NaN();  // not measured
            fmae=0;
            N=0;
            dbidx++;
        }
    }
    return fmae_vec;
}

tadah::core::t_type Analytics::calc_s_mae() const {

    tadah::core::t_type smae_vec(st.dbidx.size()-1);
    double smae=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        const bool counts = counts_in_metric(st(i), MetricQuantity::Stress);
        for (size_t x=0; counts && x<3; ++x) {
            for (size_t y=x; y<3; ++y) {
                smae += std::abs(st(i).stress(x,y) - stp(i).stress(x,y));
                N++;
            }
        }
        if (i+1==st.dbidx[dbidx+1]) {
            smae_vec(dbidx)=N ? smae/N : std::numeric_limits<double>::quiet_NaN();  // not measured
            smae=0;
            N=0;
            dbidx++;
        }
    }
    return smae_vec;
}

tadah::core::t_type Analytics::calc_e_rrmse() const {
  tadah::core::t_type errmse_vec(st.dbidx.size()-1);
  double errmse=0;
  size_t dbidx=0;
  size_t N=0;
  for (size_t i=0; i<st.size(); ++i) {
    if (st(i).energy != 0.0 && counts_in_metric(st(i), MetricQuantity::Energy)) {
      errmse += std::pow((st(i).energy - stp(i).energy)/st(i).energy,2);
      N++;
    }
    if (i+1==st.dbidx[dbidx+1]) {
      errmse_vec(dbidx)=N ? std::sqrt(errmse/N) : std::numeric_limits<double>::quiet_NaN();  // not measured
      errmse=0;
      N=0;
      dbidx++;
    }
  }
  return errmse_vec;
}

tadah::core::t_type Analytics::calc_e_rmse() const{

    tadah::core::t_type ermse_vec(st.dbidx.size()-1);
    double ermse=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        // assume stp(i).natoms()==st(i).natoms()
        if (counts_in_metric(st(i), MetricQuantity::Energy)) {
            ermse += std::pow((st(i).energy - stp(i).energy)/st(i).natoms(),2);
            N++;
        }
        if (i+1==st.dbidx[dbidx+1]) {
            ermse_vec(dbidx)=N ? std::sqrt(ermse/N) : std::numeric_limits<double>::quiet_NaN();  // not measured
            ermse=0;
            N=0;
            dbidx++;
        }
    }
    return ermse_vec;
}

tadah::core::t_type Analytics::calc_f_rmse() const{
    require_predicted_forces(st, stp, "Analytics::calc_f_rmse");

    tadah::core::t_type frmse_vec(st.dbidx.size()-1);
    double frmse=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        const bool counts = counts_in_metric(st(i), MetricQuantity::Forces);
        for (size_t a=0; counts && a<st(i).natoms(); ++a) {
            for (size_t k=0; k<3; ++k) {
                frmse += std::pow(st(i).atoms[a].force(k) - stp(i).atoms[a].force(k),2);
                N++;
            }
        }

        if (i+1==st.dbidx[dbidx+1]) {
            frmse_vec(dbidx)=N ? sqrt(frmse/N) : std::numeric_limits<double>::quiet_NaN();  // not measured
            frmse=0;
            N=0;
            dbidx++;
        }
    }
    return frmse_vec;
}

tadah::core::t_type Analytics::calc_s_rmse() const {

    tadah::core::t_type srmse_vec(st.dbidx.size()-1);
    double srmse=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        const bool counts = counts_in_metric(st(i), MetricQuantity::Stress);
        for (size_t x=0; counts && x<3; ++x) {
            for (size_t y=x; y<3; ++y) {
                srmse += std::pow(st(i).stress(x,y) - stp(i).stress(x,y),2);
                N++;
            }
        }
        if (i+1==st.dbidx[dbidx+1]) {
            srmse_vec(dbidx)=N ? std::sqrt(srmse/N) : std::numeric_limits<double>::quiet_NaN();  // not measured
            srmse=0;
            N=0;
            dbidx++;
        }
    }
    return srmse_vec;
}

tadah::core::t_type Analytics::calc_e_r_sq() const {
    const auto nseg = st.dbidx.size() - 1;
    tadah::core::t_type e_r_sq_vec(nseg);

    for (size_t dbidx = 0; dbidx < nseg; ++dbidx) {
        const size_t begin = st.dbidx[dbidx];
        const size_t end   = st.dbidx[dbidx + 1];
        assert(end >= begin);

        size_t n = 0;
        for (size_t i = begin; i < end; ++i)
            if (counts_in_metric(st(i), MetricQuantity::Energy)) ++n;
        tadah::core::t_type obs(n);
        tadah::core::t_type pred(n);

        size_t j = 0;
        for (size_t i = begin; i < end; ++i) {
            if (!counts_in_metric(st(i), MetricQuantity::Energy)) continue;
            obs(j)  = st(i).energy  / st(i).natoms();
            pred(j) = stp(i).energy / st(i).natoms();
            ++j;
        }

        e_r_sq_vec(dbidx) = n ? Statistics::r_sq(obs, pred) : std::numeric_limits<double>::quiet_NaN();  // not measured
    }

    return e_r_sq_vec;
}

tadah::core::t_type Analytics::calc_f_r_sq() const {
    require_predicted_forces(st, stp, "Analytics::calc_f_r_sq");
    const auto nseg = st.dbidx.size() - 1;
    tadah::core::t_type f_r_sq_vec(nseg);

    for (size_t dbidx = 0; dbidx < nseg; ++dbidx) {
        const size_t begin = st.dbidx[dbidx];
        const size_t end   = st.dbidx[dbidx + 1];
        size_t n_atoms = 0;
        for (size_t i = begin; i < end; ++i) {
            if (counts_in_metric(st(i), MetricQuantity::Forces)) n_atoms += st(i).natoms();
        }
        tadah::core::t_type obs(3 * n_atoms);
        tadah::core::t_type pred(3 * n_atoms);

        size_t idx = 0;
        for (size_t i = begin; i < end; ++i) {
            if (!counts_in_metric(st(i), MetricQuantity::Forces)) continue;
            for (size_t a = 0; a < st(i).natoms(); ++a) {
                for (size_t k = 0; k < 3; ++k) {
                    obs(idx) = st(i).atoms[a].force(k);
                    pred(idx) = stp(i).atoms[a].force(k);
                    idx++;
                }
            }
        }
        f_r_sq_vec(dbidx) = n_atoms ? Statistics::r_sq(obs, pred) : std::numeric_limits<double>::quiet_NaN();
    }
    return f_r_sq_vec;
}

tadah::core::t_type Analytics::calc_s_r_sq() const {
    const auto nseg = st.dbidx.size() - 1;
    tadah::core::t_type s_r_sq_vec(nseg);

    for (size_t dbidx = 0; dbidx < nseg; ++dbidx) {
        const size_t begin = st.dbidx[dbidx];
        const size_t end   = st.dbidx[dbidx + 1];
        size_t n_structs = 0;
        for (size_t i = begin; i < end; ++i)
            if (counts_in_metric(st(i), MetricQuantity::Stress)) ++n_structs;
        tadah::core::t_type obs(6 * n_structs);
        tadah::core::t_type pred(6 * n_structs);

        size_t idx = 0;
        for (size_t i = begin; i < end; ++i) {
            if (!counts_in_metric(st(i), MetricQuantity::Stress)) continue;
            for (size_t x = 0; x < 3; ++x) {
                for (size_t y = x; y < 3; ++y) {
                    obs(idx) = st(i).stress(x, y);
                    pred(idx) = stp(i).stress(x, y);
                    idx++;
                }
            }
        }
        s_r_sq_vec(dbidx) = n_structs ? Statistics::r_sq(obs, pred) : std::numeric_limits<double>::quiet_NaN();
    }
    return s_r_sq_vec;
}

} // namespace mlip
} // namespace tadah
