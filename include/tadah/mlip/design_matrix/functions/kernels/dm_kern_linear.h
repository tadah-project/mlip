#ifndef DM_KERN_LINEAR_H
#define DM_KERN_LINEAR_H

#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>
#include <tadah/models/functions/kernels/kern_linear.h>
namespace tadah {
namespace mlip {
/**
 * Linear kernel also knows as dot product kernel
 *
 * Defined for two vectors **x** and **y**:
 *
 * \f[
 * K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \mathbf{y} = \sum_i x^{(i)} y^{(i)}
 * \f]
 *
 *  @see tadah::models::Kern_Base tadah::models::BF_Linear
 */
class DM_Kern_Linear :  public DM_Kern_Base, public tadah::models::Kern_Linear {
    public:
    DM_Kern_Linear ();
    DM_Kern_Linear (const tadah::core::Context &c);

    size_t get_phi_cols(const tadah::core::Context &context) override;
    /** The descriptor layout: one column per descriptor component. */
    ColumnLayout phi_layout(const ColumnLayout &descriptor_layout,
                            const tadah::core::Context &context) const override;
    void calc_phi_energy_row(tadah::core::phi_type &Phi, size_t &row,
            const Structure &st, const StDescriptors &st_d) override;
    void calc_phi_force_rows(tadah::core::phi_type &Phi, size_t &row,
            const Structure &st, const StDescriptors &st_d) override;
    void calc_phi_stress_rows(tadah::core::phi_type &Phi, size_t &row,
            const Structure &st, const StDescriptors &st_d) override;
};
} // namespace mlip
} // namespace tadah
#endif
