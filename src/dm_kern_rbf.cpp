#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_rbf.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_Kern_RBF> DM_Kern_RBF_1( "tadah::models::Kern_RBF" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_RBF> DM_Kern_RBF_2( "tadah::models::Kern_RBF" );

DM_Kern_RBF::DM_Kern_RBF()
{}
DM_Kern_RBF::DM_Kern_RBF(const tadah::core::Context &c):
  tadah::models::Function_Base(c), 
  DM_Kern_Base(c),
  tadah::models::Kern_RBF(c)
{}
} // namespace mlip
} // namespace tadah
