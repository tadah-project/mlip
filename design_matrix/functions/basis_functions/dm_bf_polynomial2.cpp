#include "dm_bf_polynomial2.h"

Registry<DM_Function_Base>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_1( "BF_Polynomial2" );
Registry<DM_Function_Base,Config&>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_2( "BF_Polynomial2" );

DM_BF_Polynomial2::DM_BF_Polynomial2() {}
DM_BF_Polynomial2::DM_BF_Polynomial2(const Config &c): BF_Polynomial2(c)
{}
size_t DM_BF_Polynomial2::get_phi_cols(const Config &config)
{
  size_t cols = config.get<size_t>("DSIZE");
  return (cols*cols+cols)/2;
}
void DM_BF_Polynomial2::calc_phi_energy_row(phi_type &Phi,
                                            size_t &row,
                                            const double fac,
                                            const Structure &,
                                            const StDescriptors &st_d)
{
  for (size_t a=0; a<st_d.naed();++a) {
    const aed_type2& aed = st_d.get_aed(a);
    size_t b=0;
    for (size_t i=0; i<st_d.dim(); ++i) {
      for (size_t ii=i; ii<st_d.dim(); ++ii) {
        Phi(row,b++) += aed(i)*aed(ii)*fac;
      }
    }
  }
  row++;
}
void DM_BF_Polynomial2::calc_phi_force_rows(phi_type &Phi,
                                            size_t &row,
                                            const double fac,
                                            const Structure &st,
                                            const StDescriptors &st_d)
{
  for (size_t a=0; a<st.natoms(); ++a) {
    const aed_type2& aedi = st_d.get_aed(a);
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.near_neigh_idx[a][jj];
      size_t aa = st.get_nn_iindex(a,j,jj);
      const fd_type &fdji = st_d.fd[j][aa];
      const fd_type &fdij = st_d.fd[a][jj];
      const aed_type2& aedj = st_d.get_aed(j);

      for (size_t k=0; k<3; ++k) {
        size_t b=0;
        for (size_t i=0; i<fdij.rows(); ++i) {
          for (size_t ii=i; ii<fdij.rows(); ++ii) {
            Phi(row+k,b) -= fac*(fdij(i,k)*aedi(ii) + fdij(ii,k)*aedi(i)
              - fdji(i,k)*aedj(ii) - fdji(ii,k)*aedj(i));
            b++;
          }
        }
      }

    }
    row+=3;
  }
}
void DM_BF_Polynomial2::calc_phi_stress_rows(phi_type &Phi,
                                             size_t &row,
                                             const double fac[6],
                                             const Structure &st, 
                                             const StDescriptors &st_d)
{
  for (size_t a=0; a<st.natoms(); ++a) {
    const Vec3d &ri = st(a).position;
    const aed_type2& aedi = st_d.get_aed(a);
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.near_neigh_idx[a][jj];
      size_t aa = st.get_nn_iindex(a,j,jj);
      const fd_type &fdji = st_d.fd[j][aa];
      const fd_type &fdij = st_d.fd[a][jj];
      const Vec3d &rj = st.nn_pos(a,jj);
      const aed_type2& aedj = st_d.get_aed(j);
      size_t mn=0;
      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          size_t b=0;
          for (size_t i=0; i<fdij.rows(); ++i) {
            for (size_t ii=i; ii<fdij.rows(); ++ii) {
              Phi(row+mn,b) += 0.5*fac[mn]*(ri(x)-rj(x))
                *(fdij(i,y)*aedi(ii) + fdij(ii,y)*aedi(i)
                - fdji(i,y)*aedj(ii) - fdji(ii,y)*aedj(i));
              b++;
            }
          }
          mn++;
        }
      }
    }
  }
  row += 6;
}
