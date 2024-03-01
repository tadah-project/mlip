#include "analytics.h"
#include "statistics.h"

Analytics::Analytics(const StructureDB &st, const StructureDB &stp):
    st(st),
    stp(stp)
{
    if (st.size()!=stp.size())
        throw std::runtime_error("Containers differ in size.");
}

t_type Analytics::calc_e_mae() const{

    t_type emae_vec(st.dbidx.size()-1);
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

t_type Analytics::calc_f_mae() const{

    t_type fmae_vec(st.dbidx.size()-1);
    double fmae=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        // assume stp(i).natoms()==st(i).natoms()

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

t_type Analytics::calc_s_mae() const {

    t_type smae_vec(st.dbidx.size()-1);
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

t_type Analytics::calc_e_rmse() const{

    t_type ermse_vec(st.dbidx.size()-1);
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

t_type Analytics::calc_f_rmse() const{

    t_type frmse_vec(st.dbidx.size()-1);
    double frmse=0;
    size_t dbidx=0;
    size_t N=0;
    for (size_t i=0; i<st.size(); ++i) {
        // assume stp(i).natoms()==st(i).natoms()

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

t_type Analytics::calc_s_rmse() const {

    t_type srmse_vec(st.dbidx.size()-1);
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

t_type Analytics::calc_e_r_sq() const {
    t_type e_r_sq_vec(st.dbidx.size()-1);
    size_t dbidx=0;
    t_type obs(st.size(dbidx));
    t_type pred(st.size(dbidx));
    size_t idx=0;
    for (size_t i=0; i<st.size(); ++i) {
        obs(idx) = st(i).energy/st(i).natoms();
        pred(idx) = stp(i).energy/st(i).natoms();
        idx++;
        if (i+1==st.dbidx[dbidx+1]) {
            e_r_sq_vec(dbidx) = Statistics::r_sq(obs,pred);
            dbidx++;
            if (dbidx<st.dbidx.size()-2) {
                obs.resize(st.size(dbidx));
                pred.resize(st.size(dbidx));
            }
            idx=0;
        }
    }
    return e_r_sq_vec;
}

t_type Analytics::calc_f_r_sq() const {
    t_type f_r_sq_vec(st.dbidx.size()-1);
    size_t dbidx=0;
    t_type obs(3*st.calc_natoms(dbidx));
    t_type pred(3*st.calc_natoms(dbidx));
    size_t idx=0;
    for (size_t i=0; i<st.size(); ++i) {
        for (size_t a=0; a<st(i).natoms(); ++a) {
            for (size_t k=0; k<3; ++k) {
                obs(idx) = st(i).atoms[a].force(k);
                pred(idx) = stp(i).atoms[a].force(k);
                idx++;
            }
        }
        if (i+1==st.dbidx[dbidx+1]) {
            f_r_sq_vec(dbidx) = Statistics::r_sq(obs,pred);
            dbidx++;
            if (dbidx<st.dbidx.size()-2) {
                obs.resize(3*st.calc_natoms(dbidx));
                pred.resize(3*st.calc_natoms(dbidx));
            }
            idx=0;
        }
    }
    return f_r_sq_vec;
}

t_type Analytics::calc_s_r_sq() const {
    t_type s_r_sq_vec(st.dbidx.size()-1);
    size_t dbidx=0;
    t_type obs(6*st.size(dbidx));
    t_type pred(6*st.size(dbidx));
    size_t idx=0;
    for (size_t i=0; i<st.size(); ++i) {

        for (size_t x=0; x<3; ++x) {
            for (size_t y=x; y<3; ++y) {
                obs(idx) = st(i).stress(x,y);
                pred(idx) = stp(i).stress(x,y);
                idx++;
            }
        }

        if (i+1==st.dbidx[dbidx+1]) {
            s_r_sq_vec(dbidx) = Statistics::r_sq(obs,pred);
            dbidx++;
            if (dbidx<st.dbidx.size()-2) {
                obs.resize(6*st.size(dbidx));
                pred.resize(6*st.size(dbidx));
            }
            idx=0;
        }
    }
    return s_r_sq_vec;
}
