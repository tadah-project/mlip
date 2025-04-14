#include <tadah/mlip/design_matrix/functions/dm_f_all.h>
namespace tadah {
namespace mlip {

template<> tadah::core::Registry<DM_Function_Base>::Map tadah::core::Registry<DM_Function_Base>::registry{};
template<> tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Map
  tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::registry{};


tadah::core::Registry<DM_Function_Base>::Register<DM_BF_Linear> DM_BF_Linear_1( "BF_Linear" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_BF_Linear> DM_BF_Linear_2( "BF_Linear" );
tadah::core::Registry<DM_Function_Base>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_1( "BF_Polynomial2" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_BF_Polynomial2> DM_BF_Polynomial2_2( "BF_Polynomial2" );
tadah::core::Registry<DM_Function_Base>::Register<DM_Kern_Linear> DM_Kern_Linear_1( "Kern_Linear" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Linear> DM_Kern_Linear_2( "Kern_Linear" );
tadah::core::Registry<DM_Function_Base>::Register<DM_Kern_LQ> DM_Kern_LQ_1( "Kern_LQ" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_LQ> DM_Kern_LQ_2( "Kern_LQ" );
tadah::core::Registry<DM_Function_Base>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_1( "Kern_Polynomial" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Polynomial> DM_Kern_Polynomial_2( "Kern_Polynomial" );
tadah::core::Registry<DM_Function_Base>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_1( "Kern_Quadratic" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Quadratic> DM_Kern_Quadratic_2( "Kern_Quadratic" );
tadah::core::Registry<DM_Function_Base>::Register<DM_Kern_RBF> DM_Kern_RBF_1( "Kern_RBF" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_RBF> DM_Kern_RBF_2( "Kern_RBF" );
tadah::core::Registry<DM_Function_Base>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_1( "Kern_Sigmoid" );
tadah::core::Registry<DM_Function_Base,tadah::core::Context&>::Register<DM_Kern_Sigmoid> DM_Kern_Sigmoid_2( "Kern_Sigmoid" );

} // namespace mlip
} // namespace tadah
