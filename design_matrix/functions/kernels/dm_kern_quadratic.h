#ifndef DM_KERN_QUADRATIC_H
#define DM_KERN_QUADRATIC_H

#include "dm_kern_base.h"
#include "../../../../MODELS/functions/kernels/kern_quadratic.h"

class DM_Kern_Quadratic :  public DM_Kern_Base, public Kern_Quadratic {
    public:
        DM_Kern_Quadratic();
        DM_Kern_Quadratic(const Config &c);
};
#endif
