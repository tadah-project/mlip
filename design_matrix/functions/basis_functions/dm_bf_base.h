#ifndef DM_BASIS_FUNCTIONS_H
#define DM_BASIS_FUNCTIONS_H

#include <iostream>
#include "../dm_function_base.h"
#include "../../../../CORE/core_types.h"
#include "../../../../MODELS/functions/basis_functions/bf_base.h"
#include "../../../structure.h"
#include "../../../st_descriptors.h"

struct DM_BF_Base: public DM_Function_Base, public virtual BF_Base {
    virtual ~DM_BF_Base();
    virtual size_t get_phi_cols(const Config &config)=0;
    virtual void calc_phi_energy_row(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d)=0;
    virtual void calc_phi_force_rows(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d)=0;
    virtual void calc_phi_stress_rows(phi_type &Phi, size_t &row,
            const double fac[6], const Structure &st, const StDescriptors &st_d)=0;
};
#endif
