#include "tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h"
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_linear.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_Kern_Linear> DM_Kern_Linear_1( "tadah::models::Kern_Linear" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Linear> DM_Kern_Linear_2( "tadah::models::Kern_Linear" );

DM_Kern_Linear::DM_Kern_Linear() {}
DM_Kern_Linear::DM_Kern_Linear (const tadah::core::Context &c): 
  tadah::models::Function_Base(c),
  DM_Kern_Base(c),
  tadah::models::Kern_Linear(c)
{}
size_t DM_Kern_Linear::get_phi_cols(const tadah::core::Context &context)
{
  size_t cols = context.get<size_t>("DSIZE");
  return cols;
}
void DM_Kern_Linear::calc_phi_energy_row(tadah::core::phi_type &Phi, size_t &row,
                                         const Structure &, const StDescriptors &st_d)
{
  for (size_t a=0; a<st_d.naed();++a) {
    const tadah::core::aed_type &aed = st_d.get_aed(a);  // TODO
    for (size_t j=0; j<aed.size(); ++j) {
      Phi(row,j)+=aed[j];
    }
  }
  row++;
}
void DM_Kern_Linear::calc_phi_force_rows(tadah::core::phi_type &Phi, size_t &row,
                                         const Structure &st, const StDescriptors &st_d)
{
  for (size_t a=0; a<st.natoms(); ++a) {
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.nn->idx[a][jj];
      const size_t aa = st.get_nn_iindex(a,j,jj);
      for (size_t k=0; k<3; ++k) {
        tadah::core::aed_type temp = (st_d.fd[a][jj](k)-
          st_d.fd[j][aa](k));
        for (size_t d=0; d<temp.size(); ++d) {
          Phi(row+k,d) -= temp[d];
        }
      }
    }
    row+=3;
  }

}
void DM_Kern_Linear::calc_phi_stress_rows(tadah::core::phi_type &Phi, size_t &row,
                                          const Structure &st, const StDescriptors &st_d)
{
  double V_inv = 1/st.get_volume();
  for (size_t i=0; i<st.natoms(); ++i) {
    const tadah::core::Vec3d &ri = st(i).position;
    for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
      const size_t j=st.nn->idx[i][jj];
      const size_t ii = st.get_nn_iindex(i,j,jj);
      const tadah::core::fd_type &fdij = st_d.fd[i][jj];
      const tadah::core::fd_type &fdji = st_d.fd[j][ii];
      const tadah::core::Vec3d &rj = st.nn_pos(i,jj);
      size_t mn=0;
      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          tadah::core::aed_type temp = V_inv*(fdij(y)-fdji(y))*0.5*(ri(x)-rj(x));
          for (size_t d=0; d<temp.size(); ++d) {
            Phi(row+mn,d) += temp[d];
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
