#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
namespace tadah {
namespace mlip {
DM_Function_Base::DM_Function_Base() {}
DM_Function_Base::DM_Function_Base(const tadah::core::Config &c): Function_Base(c) {}
DM_Function_Base::~DM_Function_Base() {}
} // namespace mlip
} // namespace tadah
