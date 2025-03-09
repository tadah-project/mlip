#ifndef DM_KERN_POLYNOMIAL_H
#define DM_KERN_POLYNOMIAL_H

#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>
#include <tadah/models/functions/kernels/kern_polynomial.h>
namespace tadah {
namespace mlip {

class DM_Kern_Polynomial :  public DM_Kern_Base, public tadah::models::Kern_Polynomial {
    public:
        DM_Kern_Polynomial();
        DM_Kern_Polynomial(const tadah::core::Config &c);
};
} // namespace mlip
} // namespace tadah
#endif
