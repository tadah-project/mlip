#ifndef DM_KERN_BASE_H
#define DM_KERN_BASE_H

#include "tadah/core/config.h"
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/core/core_types.h>
#include <tadah/models/functions/kernels/kern_base.h>

#include <iostream>
namespace tadah {
namespace mlip {

/** \brief Abstract class to be used as a base for all kernels.
 *
 *  - b = basis vector
 *  - af = atomic energy descriptor
 *  - ff = force descriptor
 *  - all derivatives are defined wrt to the second argument
 */
class DM_Kern_Base: public DM_Function_Base, public virtual Kern_Base  {
    public:

        DM_Kern_Base();
        DM_Kern_Base(const tadah::core::Config&c);
        virtual ~DM_Kern_Base();
        virtual size_t get_phi_cols(const tadah::core::Config &config) override;
        virtual void  calc_phi_energy_row(tadah::core::phi_type &Phi, size_t &row, const Structure &st, const StDescriptors &st_d) override;
        virtual void  calc_phi_force_rows(tadah::core::phi_type &Phi, size_t &row, const Structure &st, const StDescriptors &st_d) override;
        virtual void  calc_phi_stress_rows(tadah::core::phi_type &Phi, size_t &row, const Structure &st, const StDescriptors &st_d) override;

};
} // namespace mlip
} // namespace tadah
#endif
