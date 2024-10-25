#ifndef DM_BF_LINEAR_H
#define DM_BF_LINEAR_H

#include <tadah/mlip/design_matrix/functions/basis_functions/dm_bf_base.h>
#include <tadah/models/functions/basis_functions/bf_linear.h>

struct DM_BF_Linear: public DM_BF_Base, public BF_Linear
{
    DM_BF_Linear();
    DM_BF_Linear(const Config &c);
    size_t get_phi_cols(const Config &config);
    void calc_phi_energy_row(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d);
    void calc_phi_force_rows(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d);
    void calc_phi_stress_rows(phi_type &Phi, size_t &row,
            const double fac[6], const Structure &st, const StDescriptors &st_d);
};
#endif
