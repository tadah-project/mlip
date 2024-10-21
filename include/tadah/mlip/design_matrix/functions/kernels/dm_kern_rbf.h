#ifndef DM_KERN_RBF_H
#define DM_KERN_RBF_H

#include <tadah/mlip/design_matrix/functions/kernels/dm_kern_base.h>
#include <tadah/models/functions/kernels/kern_rbf.h>

class DM_Kern_RBF :  public DM_Kern_Base, public Kern_RBF {
    public:
        DM_Kern_RBF();
        DM_Kern_RBF(const Config &c);
};
#endif
