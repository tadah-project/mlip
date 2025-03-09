#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_polynomial.h>
namespace tadah {
namespace mlip {

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_1( "tadah::models::Kern_Polynomial" );
//CONFIG::Registry<DM_Function_Base,tadah::core::Config&>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_2( "tadah::models::Kern_Polynomial" );

DM_Kern_Polynomial::DM_Kern_Polynomial()
{}
DM_Kern_Polynomial::DM_Kern_Polynomial(const tadah::core::Config &c):
  tadah::models::Function_Base(c), 
  DM_Kern_Base(c),
  tadah::models::Kern_Polynomial(c)
{}
} // namespace mlip
} // namespace tadah
