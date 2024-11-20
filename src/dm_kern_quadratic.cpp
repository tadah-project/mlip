#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_quadratic.h>

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_1( "Kern_Quadratic" );
//CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_2( "Kern_Quadratic" );

DM_Kern_Quadratic::DM_Kern_Quadratic():
  Kern_Quadratic()
{}
DM_Kern_Quadratic::DM_Kern_Quadratic(const Config &c):
  Kern_Quadratic(c)
{}
