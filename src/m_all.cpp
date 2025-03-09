#include <tadah/mlip/models/m_all.h>
#include <tadah/models/memory/IModelsWorkspaceManager.h>
#include <tadah/mlip/memory/IMLIPWorkspaceManager.h>
namespace tadah {
namespace mlip {

template<>
CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&>::Map
CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&>::registry{};

CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&>::Register<M_KRR<>> M_KRR_1("M_KRR");
CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&>::Register<M_BLR<>> M_BLR_1("M_BLR");

/*template<>*/
/*CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::models::memory::IModelsWorkspaceManager&>::Map*/
/*CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::models::memory::IModelsWorkspaceManager&>::registry{};*/
/**/
/*CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::models::memory::IModelsWorkspaceManager&>::Register<M_BLR<>> M_BLR_2("M_BLR");*/
/*CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::models::memory::IModelsWorkspaceManager&>::Register<M_KRR<>> M_KRR_2("M_KRR");*/

template<>
CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::mlip::memory::IMLIPWorkspaceManager&>::Map
CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::mlip::memory::IMLIPWorkspaceManager&>::registry{};

CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::mlip::memory::IMLIPWorkspaceManager&>::Register<M_BLR<>> M_BLR_3("M_BLR");
CONFIG::Registry<M_Tadah_Base, DM_Function_Base&, tadah::core::Config&, tadah::mlip::memory::IMLIPWorkspaceManager&>::Register<M_KRR<>> M_KRR_3("M_KRR");
} // namespace mlip
} // namespace tadah
