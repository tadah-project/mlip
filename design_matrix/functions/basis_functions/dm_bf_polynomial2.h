#ifndef DM_BF_POLYNOMIAL2_H
#define DM_BF_POLYNOMIAL2_H
#include "dm_bf_base.h"
#include "../../../../MODELS/functions/basis_functions/bf_polynomial2.h"

struct DM_BF_Polynomial2: public DM_BF_Base, public BF_Polynomial2
{
    DM_BF_Polynomial2();
    DM_BF_Polynomial2(const Config &c);
    size_t get_phi_cols(const Config &config);
    void  calc_phi_energy_row(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d);
    void  calc_phi_force_rows(phi_type &Phi, size_t &row,
            const double fac, const Structure &st, const StDescriptors &st_d);
    void  calc_phi_stress_rows(phi_type &Phi, size_t &row,
            const double fac[6], const Structure &st, const StDescriptors &st_d);
};
#endif
