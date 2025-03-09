#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_quadratic.h>
namespace tadah {
namespace mlip {

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_1( "Kern_Quadratic" );
//CONFIG::Registry<DM_Function_Base,tadah::core::Config&>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_2( "Kern_Quadratic" );

DM_Kern_Quadratic::DM_Kern_Quadratic()
{}
DM_Kern_Quadratic::DM_Kern_Quadratic(const tadah::core::Config &c):
  Function_Base(c),
  DM_Kern_Base(c),
  Kern_Quadratic(c)
{}
} // namespace mlip
} // namespace tadah
