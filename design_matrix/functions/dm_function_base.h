#ifndef DM_FUNCTION_H
#define DM_FUNCTION_H

#include "../../../CORE/config/config.h"
#include "../../../CORE/registry.h"
#include "../../../CORE/core_types.h"
#include "../../../MODELS/functions/function_base.h"
#include "../../structure.h"
#include "../../st_descriptors.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

/** Base class for Kernels and Basis Functions */
struct DM_Function_Base: public virtual Function_Base {

    // Derived classes must implement Derived() and Derived(Config)
    virtual ~DM_Function_Base() {};

    virtual size_t get_phi_cols(const Config &)=0;
    virtual void calc_phi_energy_row(phi_type &, size_t &,
            const double , const Structure &, const StDescriptors &)=0;
    virtual void calc_phi_force_rows(phi_type &, size_t &,
            const double , const Structure &, const StDescriptors &)=0;
    virtual void calc_phi_stress_rows(phi_type &, size_t &,
            const double[6], const Structure &, const StDescriptors &)=0;
};
template<> inline Registry<DM_Function_Base>::Map Registry<DM_Function_Base>::registry{};
template<> inline Registry<DM_Function_Base,Config&>::Map Registry<DM_Function_Base,Config&>::registry{};
#endif
