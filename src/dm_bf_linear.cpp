#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_linear.h>
namespace tadah {
namespace mlip {

//CONFIG::Registry<DM_Function_Base>::Register<DM_BF_Linear> DM_BF_Linear_1( "tadah::models::BF_Linear" );
//CONFIG::Registry<DM_Function_Base,tadah::core::Config&>::Register<DM_BF_Linear> DM_BF_Linear_2( "tadah::models::BF_Linear" );

DM_BF_Linear::DM_BF_Linear() {}
DM_BF_Linear::DM_BF_Linear(const tadah::core::Config &c): 
  tadah::models::Function_Base(c), 
  DM_BF_Base(c),
  tadah::models::BF_Linear(c)
{}
size_t DM_BF_Linear::get_phi_cols(const tadah::core::Config &config)
{
  size_t cols = config.get<size_t>("DSIZE");
  return cols;
}
void DM_BF_Linear::calc_phi_energy_row(tadah::core::phi_type &Phi, size_t &row,
                                       const Structure &, const StDescriptors &st_d)
{
  for (size_t i=0; i<st_d.naed(); ++i) {
    const tadah::core::aed_type &aed = st_d.get_aed(i);
    for (size_t j=0; j<aed.size(); ++j) {
      Phi(row,j)+=aed[j];
    }
  }
  row++;
}
void DM_BF_Linear::calc_phi_force_rows(tadah::core::phi_type &Phi, size_t &row,
                                       const Structure &st, const StDescriptors &st_d)
{

  for (size_t a=0; a<st.natoms(); ++a) {
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.near_neigh_idx[a][jj];
      const size_t aa = st.get_nn_iindex(a,j,jj);
      const tadah::core::fd_type &fij = st_d.fd[a][jj];
      const tadah::core::fd_type &fji = st_d.fd[j][aa];
      for (size_t k=0; k<3; ++k) {
        for (size_t d=0; d<fij.rows(); ++d) {
          Phi(row+k,d) -= (fij(d,k)-fji(d,k));
        }
      }
    }
    row+=3;
  }
}
void DM_BF_Linear::calc_phi_stress_rows(tadah::core::phi_type &Phi, size_t &row,
                                        const Structure &st, const StDescriptors &st_d)
{
  double V_inv = 1/st.get_volume();
  for (size_t i=0; i<st.natoms(); ++i) {
    const tadah::core::Vec3d &ri = st(i).position;
    for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
      const size_t j=st.near_neigh_idx[i][jj];
      const size_t ii = st.get_nn_iindex(i,j,jj);
      const tadah::core::fd_type &fdij = st_d.fd[i][jj];
      const tadah::core::fd_type &fdji = st_d.fd[j][ii];
      const tadah::core::Vec3d &rj = st.nn_pos(i,jj);
      size_t mn=0;
      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          for (size_t d=0; d<fdij.rows(); ++d) {
            Phi(row+mn,d) += V_inv*(fdij(d,y)-fdji(d,y))*0.5*(ri(x)-rj(x));
          }
          mn++;
        }
      }
    }
  }
  row += 6;
}
} // namespace mlip
} // namespace tadah
