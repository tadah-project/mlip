#include "tadah/core/context.h"
#include "tadah/mlip/design_matrix/functions/dm_function_base.h"
#include "tadah/models/functions/kernels/kern_base.h"
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>

#include <stdexcept>
#include <string>
namespace tadah {
namespace mlip {

DM_Kern_Base::~DM_Kern_Base() {}
DM_Kern_Base::DM_Kern_Base() {}
DM_Kern_Base::DM_Kern_Base(const tadah::core::Context &c): 
  tadah::models::Function_Base(c), 
  tadah::models::Kern_Base(c), 
  DM_Function_Base(c) {}
size_t DM_Kern_Base::get_phi_cols(const tadah::core::Context &context)
{
  return weights_count(context.get<size_t>("DSIZE"));   // basis.cols()
}
void  DM_Kern_Base::calc_phi_energy_row(tadah::core::phi_type &Phi, size_t &row,
                                        const Structure &, const StDescriptors &st_d)
{
  for (size_t a=0; a<st_d.naed();++a) {
    for (size_t b=0; b<basis.cols(); ++b) {
      Phi(row,b) += (*this)(basis.col(b),st_d.get_aed(a));
    }
  }
  row++;
}
void  DM_Kern_Base::calc_phi_force_rows(tadah::core::phi_type &Phi, size_t &row,
                                        const Structure &st, const StDescriptors &st_d) {
  for (size_t i=0; i<st.natoms(); ++i) {
    const tadah::core::aed_type& aedi = st_d.get_aed(i);
    for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
      size_t j=st.nn->idx[i][jj];
      size_t ii = st.get_nn_iindex(i,j,jj);
      const tadah::core::fd_type &fdji = st_d.fd[j][ii];
      const tadah::core::fd_type &fdij = st_d.fd[i][jj];
      const tadah::core::aed_type& aedj = st_d.get_aed(j);
      for (size_t b=0; b<basis.cols(); ++b) {
        for (size_t k=0; k<3; ++k) {
          Phi(row+k,b) -= ((*this).prime(basis.col(b), aedi,fdij(k)) -
            (*this).prime(basis.col(b),aedj,fdji(k)));
        }
      }
    }
    row+=3;
  }
}
void  DM_Kern_Base::calc_phi_stress_rows(tadah::core::phi_type &Phi, size_t &row,
                                         const Structure &st, const StDescriptors &st_d)
{
  double V_inv = 1/st.get_volume();
  for (size_t i=0; i<st.natoms(); ++i) {
    const tadah::core::Vec3d &ri = st(i).position;
    const tadah::core::aed_type& aedi = st_d.get_aed(i);
    for (size_t jj=0; jj<st_d.fd[i].size(); ++jj) {
      size_t j=st.nn->idx[i][jj];
      size_t ii = st.get_nn_iindex(i,j,jj);
      const tadah::core::fd_type &fdji = st_d.fd[j][ii];
      const tadah::core::fd_type &fdij = st_d.fd[i][jj];
      const tadah::core::aed_type& aedj = st_d.get_aed(j);
      const tadah::core::Vec3d &rj = st.nn_pos(i,jj);
      size_t mn=0;
      for (size_t x=0; x<3; ++x) {
        for (size_t y=x; y<3; ++y) {
          for (size_t b=0; b<basis.cols(); ++b) {
            Phi(row+mn,b) += V_inv*0.5*(ri(x)-rj(x))*
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
ColumnLayout DM_Kern_Base::phi_layout(const ColumnLayout &descriptor_layout,
                                      const tadah::core::Context &) const
{
  // Column b is K(basis vector b, aed) summed over the atoms: one block of
  // the kernel's basis vectors, which are compared with the descriptors.
  if (basis.cols() == 0)
    throw std::runtime_error(
        "kernel " + get_label() + " has no basis vectors (SBASIS, BASIS): "
        "the design-matrix columns of a non-linear kernel are its basis "
        "vectors");
  if (basis.rows() != descriptor_layout.total_cols())
    throw std::runtime_error(
        "the basis vectors of kernel " + get_label() + " have " +
        std::to_string(basis.rows()) + " components, and the descriptors "
        "have " + std::to_string(descriptor_layout.total_cols()) +
        " (DSIZE): the basis (BASIS) does not belong to these descriptors");
  ColumnBlock b;
  b.descriptor_label = get_label();
  b.family = "kernel";
  b.col_offset = 0;
  b.col_count = basis.cols();
  ColumnLayout L;
  L.blocks.push_back(b);
  return L;
}
} // namespace mlip
} // namespace tadah
