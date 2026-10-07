#include "tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h"
#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_polynomial2.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_1( "tadah::models::BF_Polynomial2" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_2( "tadah::models::BF_Polynomial2" );

DM_BF_Polynomial2::DM_BF_Polynomial2() {}
DM_BF_Polynomial2::DM_BF_Polynomial2(const tadah::core::Context &c): 
  tadah::models::Function_Base(c),
  DM_BF_Base(c),
  tadah::models::BF_Polynomial2(c)
{}
size_t DM_BF_Polynomial2::get_phi_cols(const tadah::core::Context &context)
{
  size_t cols = context.get<size_t>("DSIZE");
  return (cols*cols+cols)/2;
}
void DM_BF_Polynomial2::calc_phi_energy_row(tadah::core::phi_type &Phi,
                                            size_t &row,
                                            const Structure &,
                                            const StDescriptors &st_d)
{
  for (size_t a=0; a<st_d.naed();++a) {
    const tadah::core::aed_type& aed = st_d.get_aed(a);
    size_t b=0;
    for (size_t i=0; i<st_d.dim(); ++i) {
      for (size_t ii=i; ii<st_d.dim(); ++ii) {
        Phi(row,b++) += aed(i)*aed(ii);
      }
    }
  }
  row++;
}
void DM_BF_Polynomial2::calc_phi_force_rows(tadah::core::phi_type &Phi,
                                            size_t &row,
                                            const Structure &st,
                                            const StDescriptors &st_d)
{
  for (size_t a=0; a<st.natoms(); ++a) {
    const tadah::core::aed_type& aedi = st_d.get_aed(a);
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.nn->idx[a][jj];
      size_t aa = st.get_nn_iindex(a,j,jj);
      const tadah::core::fd_type &fdji = st_d.fd[j][aa];
      const tadah::core::fd_type &fdij = st_d.fd[a][jj];
      const tadah::core::aed_type& aedj = st_d.get_aed(j);

      for (size_t k=0; k<3; ++k) {
        size_t b=0;
        for (size_t i=0; i<fdij.rows(); ++i) {
          for (size_t ii=i; ii<fdij.rows(); ++ii) {
            Phi(row+k,b) -= (fdij(i,k)*aedi(ii) + fdij(ii,k)*aedi(i)
              - fdji(i,k)*aedj(ii) - fdji(ii,k)*aedj(i));
            b++;
          }
        }
      }

    }
    row+=3;
  }
}
void DM_BF_Polynomial2::calc_phi_stress_rows(tadah::core::phi_type &Phi,
                                             size_t &row,
                                             const Structure &st, 
                                             const StDescriptors &st_d)
{
  double V_inv = 1/st.get_volume();
  for (size_t a=0; a<st.natoms(); ++a) {
    const tadah::core::Vec3d &ri = st(a).position;
    const tadah::core::aed_type& aedi = st_d.get_aed(a);
    for (size_t jj=0; jj<st_d.fd[a].size(); ++jj) {
      const size_t j=st.nn->idx[a][jj];
      size_t aa = st.get_nn_iindex(a,j,jj);
      const tadah::core::fd_type &fdji = st_d.fd[j][aa];
      const tadah::core::fd_type &fdij = st_d.fd[a][jj];
      const tadah::core::Vec3d &rj = st.nn_pos(a,jj);
      const tadah::core::aed_type& aedj = st_d.get_aed(j);
      size_t mn=0;
      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          size_t b=0;
          for (size_t i=0; i<fdij.rows(); ++i) {
            for (size_t ii=i; ii<fdij.rows(); ++ii) {
              Phi(row+mn,b) += V_inv*0.5*(ri(x)-rj(x))
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
ColumnLayout DM_BF_Polynomial2::phi_layout(const ColumnLayout &descriptor_layout,
                                           const tadah::core::Context &) const
{
  // Column (i, ii), i <= ii, is aed(i)*aed(ii): the products cross the
  // descriptor blocks, so the function owns all D(D+1)/2 columns.
  const size_t d = descriptor_layout.total_cols();
  ColumnBlock b;
  b.descriptor_label = get_label();
  b.family = "BF";
  b.col_offset = 0;
  b.col_count = (d*d+d)/2;
  ColumnLayout L;
  L.blocks.push_back(b);
  return L;
}
} // namespace mlip
} // namespace tadah
