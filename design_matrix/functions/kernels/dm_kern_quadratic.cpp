#include "dm_kern_quadratic.h"

Registry<DM_Function_Base>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_1( "Kern_Quadratic" );
Registry<DM_Function_Base,Config&>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_2( "Kern_Quadratic" );

DM_Kern_Quadratic::DM_Kern_Quadratic():
    Kern_Quadratic()
{}
DM_Kern_Quadratic::DM_Kern_Quadratic(const Config &c):
    Kern_Quadratic(c)
{}
