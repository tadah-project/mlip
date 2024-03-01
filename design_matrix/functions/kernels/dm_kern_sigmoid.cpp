#include "dm_kern_sigmoid.h"

Registry<DM_Function_Base>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_1( "Kern_Sigmoid" );
Registry<DM_Function_Base,Config&>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_2( "Kern_Sigmoid" );

DM_Kern_Sigmoid::DM_Kern_Sigmoid():
    Kern_Sigmoid()
{}
DM_Kern_Sigmoid::DM_Kern_Sigmoid(const Config &c):
    Kern_Sigmoid(c)
{}
