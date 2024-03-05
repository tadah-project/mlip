#include "dm_f_all.h"

template<> CONFIG::Registry<DM_Function_Base>::Map CONFIG::Registry<DM_Function_Base>::registry{};
template<> CONFIG::Registry<DM_Function_Base,Config&>::Map CONFIG::Registry<DM_Function_Base,Config&>::registry{};


CONFIG::Registry<DM_Function_Base>::Register<DM_BF_Linear> DM_BF_Linear_1( "BF_Linear" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_BF_Linear> DM_BF_Linear_2( "BF_Linear" );
CONFIG::Registry<DM_Function_Base>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_1( "BF_Polynomial2" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_2( "BF_Polynomial2" );
CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Linear> DM_Kern_Linear_1( "Kern_Linear" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_Linear> DM_Kern_Linear_2( "Kern_Linear" );
CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_LQ> DM_Kern_LQ_1( "Kern_LQ" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_LQ> DM_Kern_LQ_2( "Kern_LQ" );
CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_1( "Kern_Polynomial" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_2( "Kern_Polynomial" );
CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_1( "Kern_Quadratic" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_2( "Kern_Quadratic" );
CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_RBF> DM_Kern_RBF_1( "Kern_RBF" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_RBF> DM_Kern_RBF_2( "Kern_RBF" );
CONFIG::Registry<DM_Function_Base>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_1( "Kern_Sigmoid" );
CONFIG::Registry<DM_Function_Base,Config&>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_2( "Kern_Sigmoid" );
