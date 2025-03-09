#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_sigmoid.h>
namespace tadah {
namespace mlip {

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_1( "Kern_Sigmoid" );
//CONFIG::Registry<DM_Function_Base,tadah::core::Config&>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_2( "Kern_Sigmoid" );

DM_Kern_Sigmoid::DM_Kern_Sigmoid()
{}
DM_Kern_Sigmoid::DM_Kern_Sigmoid(const tadah::core::Config &c):
  Function_Base(c), 
  DM_Kern_Base(c),
  Kern_Sigmoid(c)
{}
} // namespace mlip
} // namespace tadah
