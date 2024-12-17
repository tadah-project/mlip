#ifndef DM_BASIS_FUNCTIONS_H
#define DM_BASIS_FUNCTIONS_H

#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/core/core_types.h>
#include <tadah/models/functions/basis_functions/bf_base.h>

#include <iostream>

struct DM_BF_Base: public DM_Function_Base, public virtual BF_Base {
   
    DM_BF_Base();
    DM_BF_Base(const Config &c);
    virtual ~DM_BF_Base();
    // virtual size_t get_phi_cols(const Config &config)=0;
    // virtual void calc_phi_energy_row(phi_type &Phi, size_t &row,
    //         const double fac, const Structure &st, const StDescriptors &st_d)=0;
    // virtual void calc_phi_force_rows(phi_type &Phi, size_t &row,
    //         const double fac, const Structure &st, const StDescriptors &st_d)=0;
    // virtual void calc_phi_stress_rows(phi_type &Phi, size_t &row,
    //         const double fac[6], const Structure &st, const StDescriptors &st_d)=0;
};
#endif
