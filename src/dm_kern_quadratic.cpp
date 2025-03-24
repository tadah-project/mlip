#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_quadratic.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_1( "tadah::models::Kern_Quadratic" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_2( "tadah::models::Kern_Quadratic" );

DM_Kern_Quadratic::DM_Kern_Quadratic()
{}
DM_Kern_Quadratic::DM_Kern_Quadratic(const tadah::core::Context &c):
  tadah::models::Function_Base(c),
  DM_Kern_Base(c),
  tadah::models::Kern_Quadratic(c)
{}
} // namespace mlip
} // namespace tadah
