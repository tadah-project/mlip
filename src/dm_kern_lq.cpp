#include "tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h"
#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_lq.h>

//CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_LQ> DM_Kern_LQ_1( "Kern_LQ" );
//CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_LQ> DM_Kern_LQ_2( "Kern_LQ" );

DM_Kern_LQ::DM_Kern_LQ()
{}
DM_Kern_LQ::DM_Kern_LQ(const Config &c):
  Function_Base(c), 
  DM_Kern_Base(c),
  Kern_LQ(c)
{}
