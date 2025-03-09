#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_rbf.h>
namespace tadah {
namespace mlip {

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_RBF> DM_Kern_RBF_1( "Kern_RBF" );
//CONFIG::Registry<DM_Function_Base,tadah::core::Config&>::Register<DM_Kern_RBF> DM_Kern_RBF_2( "Kern_RBF" );

DM_Kern_RBF::DM_Kern_RBF()
{}
DM_Kern_RBF::DM_Kern_RBF(const tadah::core::Config &c):
  Function_Base(c), 
  DM_Kern_Base(c),
  Kern_RBF(c)
{}
} // namespace mlip
} // namespace tadah
