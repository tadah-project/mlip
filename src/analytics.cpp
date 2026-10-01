#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/statistics.h>
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
        emae += std::abs(st(i).energy - stp(i).energy)/st(i).natoms();
        N++;
        if (i+1==st.dbidx[dbidx+1]) {
            emae_vec(dbidx)=emae/N;
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
        for (size_t a=0; a<st(i).natoms(); ++a) {
            for (size_t k=0; k<3; ++k) {
                fmae += std::abs(st(i).atoms[a].force(k) - stp(i).atoms[a].force(k));
                N++;
            }
        }

        if (i+1==st.dbidx[dbidx+1]) {
            fmae_vec(dbidx)=fmae/N;
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
        for (size_t x=0; x<3; ++x) {
            for (size_t y=x; y<3; ++y) {
                smae += std::abs(st(i).stress(x,y) - stp(i).stress(x,y));
                N++;
            }
        }
        if (i+1==st.dbidx[dbidx+1]) {
            smae_vec(dbidx)=smae/N;
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
    if (st(i).energy != 0.0) {
      errmse += std::pow((st(i).energy - stp(i).energy)/st(i).energy,2);
      N++;
    }
    if (i+1==st.dbidx[dbidx+1]) {
      errmse_vec(dbidx)=std::sqrt(errmse/N);
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
        ermse += std::pow((st(i).energy - stp(i).energy)/st(i).natoms(),2);
        N++;
        if (i+1==st.dbidx[dbidx+1]) {
            ermse_vec(dbidx)=std::sqrt(ermse/N);
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
        for (size_t a=0; a<st(i).natoms(); ++a) {
            for (size_t k=0; k<3; ++k) {
                frmse += std::pow(st(i).atoms[a].force(k) - stp(i).atoms[a].force(k),2);
                N++;
            }
        }

        if (i+1==st.dbidx[dbidx+1]) {
            frmse_vec(dbidx)=sqrt(frmse/N);
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
        for (size_t x=0; x<3; ++x) {
            for (size_t y=x; y<3; ++y) {
                srmse += std::pow(st(i).stress(x,y) - stp(i).stress(x,y),2);
                N++;
            }
        }
        if (i+1==st.dbidx[dbidx+1]) {
            srmse_vec(dbidx)=std::sqrt(srmse/N);
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

        const size_t n = end - begin;
        tadah::core::t_type obs(n);
        tadah::core::t_type pred(n);

        for (size_t j = 0; j < n; ++j) {
            const size_t i = begin + j;
            obs(j)  = st(i).energy  / st(i).natoms();
            pred(j) = stp(i).energy / st(i).natoms();
        }

        e_r_sq_vec(dbidx) = Statistics::r_sq(obs, pred);
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
            n_atoms += st(i).natoms();
        }
        tadah::core::t_type obs(3 * n_atoms);
        tadah::core::t_type pred(3 * n_atoms);

        size_t idx = 0;
        for (size_t i = begin; i < end; ++i) {
            for (size_t a = 0; a < st(i).natoms(); ++a) {
                for (size_t k = 0; k < 3; ++k) {
                    obs(idx) = st(i).atoms[a].force(k);
                    pred(idx) = stp(i).atoms[a].force(k);
                    idx++;
                }
            }
        }
        f_r_sq_vec(dbidx) = Statistics::r_sq(obs, pred);
    }
    return f_r_sq_vec;
}

tadah::core::t_type Analytics::calc_s_r_sq() const {
    const auto nseg = st.dbidx.size() - 1;
    tadah::core::t_type s_r_sq_vec(nseg);

    for (size_t dbidx = 0; dbidx < nseg; ++dbidx) {
        const size_t begin = st.dbidx[dbidx];
        const size_t end   = st.dbidx[dbidx + 1];
        const size_t n_structs = end - begin;
        tadah::core::t_type obs(6 * n_structs);
        tadah::core::t_type pred(6 * n_structs);

        size_t idx = 0;
        for (size_t i = begin; i < end; ++i) {
            for (size_t x = 0; x < 3; ++x) {
                for (size_t y = x; y < 3; ++y) {
                    obs(idx) = st(i).stress(x, y);
                    pred(idx) = stp(i).stress(x, y);
                    idx++;
                }
            }
        }
        s_r_sq_vec(dbidx) = Statistics::r_sq(obs, pred);
    }
    return s_r_sq_vec;
}

} // namespace mlip
} // namespace tadah
