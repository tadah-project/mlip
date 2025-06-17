#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_polynomial.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_1( "tadah::models::Kern_Polynomial" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_2( "tadah::models::Kern_Polynomial" );

DM_Kern_Polynomial::DM_Kern_Polynomial()
{}
DM_Kern_Polynomial::DM_Kern_Polynomial(const tadah::core::Context &c):
  tadah::models::Function_Base(c), 
  DM_Kern_Base(c),
  tadah::models::Kern_Polynomial(c)
{}
} // namespace mlip
} // namespace tadah
