#ifndef DM_KERN_SIGMOID_H
#define DM_KERN_SIGMOID_H

#include "dm_kern_base.h"
#include "../../../../MODELS/functions/kernels/kern_sigmoid.h"

class DM_Kern_Sigmoid :  public DM_Kern_Base, public Kern_Sigmoid {
    public:
        DM_Kern_Sigmoid();
        DM_Kern_Sigmoid(const Config &c);
};
#endif
