#ifndef DM_KERN_SIGMOID_H
#define DM_KERN_SIGMOID_H

#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>
#include <tadah/models/functions/kernels/kern_sigmoid.h>
namespace tadah {
namespace mlip {

class DM_Kern_Sigmoid :  public DM_Kern_Base, public tadah::models::Kern_Sigmoid {
    public:
        DM_Kern_Sigmoid();
        DM_Kern_Sigmoid(const tadah::core::Config &c);
};
} // namespace mlip
} // namespace tadah
#endif
