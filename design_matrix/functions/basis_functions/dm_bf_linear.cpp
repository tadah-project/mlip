#include "dm_bf_linear.h"

Registry<DM_Function_Base>::Register<DM_BF_Linear> DM_BF_Linear_1( "BF_Linear" );
Registry<DM_Function_Base,Config&>::Register<DM_BF_Linear> DM_BF_Linear_2( "BF_Linear" );

DM_BF_Linear::DM_BF_Linear() {}
DM_BF_Linear::DM_BF_Linear(const Config &c): BF_Linear(c)
{}
size_t DM_BF_Linear::get_phi_cols(const Config &config)
{
  size_t cols = config.get<size_t>("DSIZE");
  return cols;
}
void DM_BF_Linear::calc_phi_energy_row(phi_type &Phi, size_t &row,
                                       const double fac, const Structure &, const StDescriptors &st_d)
{
  for (size_t i=0; i<st_d.naed(); ++i) {
    const aed_type2 &aed = st_d.get_aed(i);
    for (size_t j=0; j<aed.size(); ++j) {
      Phi(row,j)+=aed[j]*fac;
    }
  }
  row++;
}
void DM_BF_Linear::calc_phi_force_rows(phi_type &Phi, size_t &row,
                                       const double fac, const Structure &st, const StDescriptors &st_d)
{

  for (size_t a=0; a<st.natoms(); ++a) {
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.near_neigh_idx[a][jj];
      const size_t aa = st.get_nn_iindex(a,j,jj);
      const fd_type &fij = st_d.fd[a][jj];
      const fd_type &fji = st_d.fd[j][aa];
      for (size_t k=0; k<3; ++k) {
        for (size_t d=0; d<fij.rows(); ++d) {
          Phi(row+k,d) -= fac*(fij(d,k)-fji(d,k));
        }
      }
    }
    row+=3;
  }
}
void DM_BF_Linear::calc_phi_stress_rows(phi_type &Phi, size_t &row,
                                        const double fac[6], const Structure &st, const StDescriptors &st_d)
{
  for (size_t i=0; i<st.natoms(); ++i) {
    const Vec3d &ri = st(i).position;
    for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
      const size_t j=st.near_neigh_idx[i][jj];
      const size_t ii = st.get_nn_iindex(i,j,jj);
      const fd_type &fdij = st_d.fd[i][jj];
      const fd_type &fdji = st_d.fd[j][ii];
      const Vec3d &rj = st.nn_pos(i,jj);
      size_t mn=0;
      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          for (size_t d=0; d<fdij.rows(); ++d) {
            Phi(row+mn,d) += (fdij(d,y)-fdji(d,y))*0.5*fac[mn]*(ri(x)-rj(x));
          }
          mn++;
        }
      }
    }
  }
  row += 6;
}
