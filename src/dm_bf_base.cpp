#include "tadah/mlip/design_matrix/functions/dm_function_base.h"
#include "tadah/models/functions/basis_functions/bf_base.h"
#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_base.h>
namespace tadah {
namespace mlip {
DM_BF_Base::DM_BF_Base() {}
DM_BF_Base::DM_BF_Base(const tadah::core::Context &c):
  tadah::models::Function_Base(c), 
  tadah::models::BF_Base(c),
  DM_Function_Base(c) {}
DM_BF_Base::~DM_BF_Base() {}
} // namespace mlip
} // namespace tadah
