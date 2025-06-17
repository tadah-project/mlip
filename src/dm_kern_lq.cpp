#include "tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h"
#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_lq.h>
namespace tadah {
namespace mlip {

//context::Registry<DM_Function_Base>::Register<DM_Kern_LQ> DM_Kern_LQ_1( "tadah::models::Kern_LQ" );
//context::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_LQ> DM_Kern_LQ_2( "tadah::models::Kern_LQ" );

DM_Kern_LQ::DM_Kern_LQ()
{}
DM_Kern_LQ::DM_Kern_LQ(const tadah::core::Context &c):
  tadah::models::Function_Base(c), 
  DM_Kern_Base(c),
  tadah::models::Kern_LQ(c)
{}
} // namespace mlip
} // namespace tadah
