#ifndef DM_KERN_BASE_H
#define DM_KERN_BASE_H

#include <iostream>
#include "../dm_function_base.h"
#include "../../../../CORE/core_types.h"
#include "../../../../MODELS/functions/kernels/kern_base.h"
#include "../../../structure.h"
#include "../../../st_descriptors.h"

/** \brief Abstract class to be used as a base for all kernels.
 *
 *  - b = basis vector
 *  - af = atomic energy descriptor
 *  - ff = force descriptor
 *  - all derivatives are defined wrt to the second argument
 */
class DM_Kern_Base: public DM_Function_Base, public virtual Kern_Base {
    public:

        virtual ~DM_Kern_Base();
        virtual size_t get_phi_cols(const Config &config);
        virtual void  calc_phi_energy_row(phi_type &Phi, size_t &row, const double fac,
                const Structure &st, const StDescriptors &st_d);
        virtual void  calc_phi_force_rows(phi_type &Phi, size_t &row, const double fac,
                const Structure &st, const StDescriptors &st_d);
        virtual void  calc_phi_stress_rows(phi_type &Phi, size_t &row, const double fac[6],
                const Structure &st, const StDescriptors &st_d);

};
#endif
