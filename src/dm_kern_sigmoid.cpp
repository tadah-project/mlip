#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_sigmoid.h>

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_1( "Kern_Sigmoid" );
//CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_2( "Kern_Sigmoid" );

DM_Kern_Sigmoid::DM_Kern_Sigmoid()
{}
DM_Kern_Sigmoid::DM_Kern_Sigmoid(const Config &c):
  DM_Kern_Base(c),
  Kern_Sigmoid(c)
{}
