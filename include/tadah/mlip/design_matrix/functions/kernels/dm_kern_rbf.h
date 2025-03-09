#ifndef DM_KERN_RBF_H
#define DM_KERN_RBF_H

#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>
#include <tadah/models/functions/kernels/kern_rbf.h>
namespace tadah {
namespace mlip {

class DM_Kern_RBF :  public DM_Kern_Base, public tadah::models::Kern_RBF {
    public:
        DM_Kern_RBF();
        DM_Kern_RBF(const tadah::core::Config &c);
};
} // namespace mlip
} // namespace tadah
#endif
