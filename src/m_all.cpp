#include <tadah/mlip/models/m_all.h>
#include <tadah/models/memory/imodels_workspace_manager.h>
#include <tadah/mlip/memory/imlip_workspace_manager.h>
namespace tadah {
namespace mlip {

template<>
tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&>::Map
tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&>::registry{};

tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&>::Register<M_KRR<>> M_KRR_1("M_KRR");
tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&>::Register<M_BLR<>> M_BLR_1("M_BLR");

/*template<>*/
/*tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::models::memory::IModelsWorkspaceManager&>::Map*/
/*tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::models::memory::IModelsWorkspaceManager&>::registry{};*/
/**/
/*tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::models::memory::IModelsWorkspaceManager&>::Register<M_BLR<>> M_BLR_2("M_BLR");*/
/*tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::models::memory::IModelsWorkspaceManager&>::Register<M_KRR<>> M_KRR_2("M_KRR");*/

template<>
tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::mlip::memory::IMLIPWorkspaceManager&>::Map
tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::mlip::memory::IMLIPWorkspaceManager&>::registry{};

tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::mlip::memory::IMLIPWorkspaceManager&>::Register<M_BLR<>> M_BLR_3("M_BLR");
tadah::core::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Context&, tadah::mlip::memory::IMLIPWorkspaceManager&>::Register<M_KRR<>> M_KRR_3("M_KRR");
} // namespace mlip
} // namespace tadah
