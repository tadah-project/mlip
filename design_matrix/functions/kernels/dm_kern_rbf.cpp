#include "dm_kern_rbf.h"

Registry<DM_Function_Base>::Register<DM_Kern_RBF> DM_Kern_RBF_1( "Kern_RBF" );
Registry<DM_Function_Base,Config&>::Register<DM_Kern_RBF> DM_Kern_RBF_2( "Kern_RBF" );

DM_Kern_RBF::DM_Kern_RBF():
    Kern_RBF()
{}
DM_Kern_RBF::DM_Kern_RBF(const Config &c):
    Kern_RBF(c)
{}
