#ifndef DM_KERN_QUADRATIC_H
#define DM_KERN_QUADRATIC_H

#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>
#include <tadah/models/functions/kernels/kern_quadratic.h>
namespace tadah {
namespace mlip {

class DM_Kern_Quadratic :  public DM_Kern_Base, public tadah::models::Kern_Quadratic {
    public:
        DM_Kern_Quadratic();
        DM_Kern_Quadratic(const tadah::core::Config &c);
};
} // namespace mlip
} // namespace tadah
#endif
