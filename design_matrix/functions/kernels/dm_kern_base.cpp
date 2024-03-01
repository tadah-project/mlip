#include "dm_kern_base.h"

DM_Kern_Base::~DM_Kern_Base() {}
size_t DM_Kern_Base::get_phi_cols(const Config &)
{
    return basis.cols();
}
void  DM_Kern_Base::calc_phi_energy_row(phi_type &Phi, size_t &row, const double fac,
        const Structure &, const StDescriptors &st_d)
{
    for (size_t a=0; a<st_d.naed();++a) {
        for (size_t b=0; b<basis.cols(); ++b) {
            Phi(row,b) += (*this)(basis.col(b),st_d.get_aed(a))*fac;
        }
    }
    row++;
    //Phi.row(row++) *= fac;
}
void  DM_Kern_Base::calc_phi_force_rows(phi_type &Phi, size_t &row, const double fac,
        const Structure &st, const StDescriptors &st_d) {
    for (size_t i=0; i<st.natoms(); ++i) {
        const aed_type2& aedi = st_d.get_aed(i);
        for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
            size_t j=st.near_neigh_idx[i][jj];
            size_t ii = st.get_nn_iindex(i,j,jj);
            const fd_type &fdji = st_d.fd[j][ii];
            const fd_type &fdij = st_d.fd[i][jj];
            const aed_type2& aedj = st_d.get_aed(j);
            for (size_t b=0; b<basis.cols(); ++b) {
                for (size_t k=0; k<3; ++k) {
                    Phi(row+k,b) -= fac*((*this).prime(basis.col(b), aedi,fdij(k)) -
                            (*this).prime(basis.col(b),aedj,fdji(k)));
                }
            }
        }
        row+=3;
    }
}
void  DM_Kern_Base::calc_phi_stress_rows(phi_type &Phi, size_t &row, const double fac[6],
        const Structure &st, const StDescriptors &st_d)
{
    for (size_t i=0; i<st.natoms(); ++i) {
        const Vec3d &ri = st(i).position;
        const aed_type2& aedi = st_d.get_aed(i);
        for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
            size_t j=st.near_neigh_idx[i][jj];
            size_t ii = st.get_nn_iindex(i,j,jj);
            const fd_type &fdji = st_d.fd[j][ii];
            const fd_type &fdij = st_d.fd[i][jj];
            const aed_type2& aedj = st_d.get_aed(j);
            const Vec3d &rj = st.nn_pos(i,jj);
            size_t mn=0;
            for (size_t x=0; x<3; ++x) {
                for (size_t y=x; y<3; ++y) {
                    for (size_t b=0; b<basis.cols(); ++b) {
                        Phi(row+mn,b) += 0.5*fac[mn]*(ri(x)-rj(x))*
                            ((*this).prime(basis.col(b),aedi,fdij(y)) -
                             (*this).prime(basis.col(b),aedj,fdji(y)));
                    }
                    mn++;
                }
            }
        }
    }
    row+=6;
}
