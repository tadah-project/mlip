#ifndef DM_KERN_LINEAR_H
#define DM_KERN_LINEAR_H

#include "dm_kern_base.h"
#include "../../../../MODELS/functions/kernels/kern_linear.h"
/**
 * Linear kernel also knows as dot product kernel
 *
 * Defined for two vectors **x** and **y**:
 *
 * \f[
 * K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \mathbf{y} = \sum_i x^{(i)} y^{(i)}
 * \f]
 *
 *  @see Kern_Base BF_Linear
 */
class DM_Kern_Linear :  public DM_Kern_Base, public Kern_Linear {
    public:
    DM_Kern_Linear ();
    DM_Kern_Linear (const Config &c);

    size_t get_phi_cols(const Config &config);
    void calc_phi_energy_row(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d);
    void calc_phi_force_rows(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d);
    void calc_phi_stress_rows(phi_type &Phi, size_t &row,
            const double fac[6], const Structure &st, const StDescriptors &st_d);
};
#endif
