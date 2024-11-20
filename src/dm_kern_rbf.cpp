#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_rbf.h>

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_RBF> DM_Kern_RBF_1( "Kern_RBF" );
//CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_RBF> DM_Kern_RBF_2( "Kern_RBF" );

DM_Kern_RBF::DM_Kern_RBF():
  Kern_RBF()
{}
DM_Kern_RBF::DM_Kern_RBF(const Config &c):
  Kern_RBF(c)
{}
