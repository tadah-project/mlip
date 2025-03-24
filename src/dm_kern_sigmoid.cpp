#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_sigmoid.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_1( "tadah::models::Kern_Sigmoid" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_2( "tadah::models::Kern_Sigmoid" );

DM_Kern_Sigmoid::DM_Kern_Sigmoid()
{}
DM_Kern_Sigmoid::DM_Kern_Sigmoid(const tadah::core::Context &c):
  tadah::models::Function_Base(c), 
  DM_Kern_Base(c),
  tadah::models::Kern_Sigmoid(c)
{}
} // namespace mlip
} // namespace tadah
