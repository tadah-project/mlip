#include "tadah/models/functions/function_base.h"
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
DM_Function_Base::DM_Function_Base() {}
DM_Function_Base::DM_Function_Base(const Config &c): Function_Base(c) {}
DM_Function_Base::~DM_Function_Base() {}
