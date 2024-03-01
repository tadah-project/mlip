#include "dm_kern_lq.h"

Registry<DM_Function_Base>::Register<DM_Kern_LQ> DM_Kern_LQ_1( "Kern_LQ" );
Registry<DM_Function_Base,Config&>::Register<DM_Kern_LQ> DM_Kern_LQ_2( "Kern_LQ" );

DM_Kern_LQ::DM_Kern_LQ():
    Kern_LQ()
{}
DM_Kern_LQ::DM_Kern_LQ(const Config &c):
    Kern_LQ(c)
{}
