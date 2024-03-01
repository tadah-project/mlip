#ifndef DM_KERN_LQ_H
#define DM_KERN_LQ_H

#include "dm_kern_base.h"
#include "../../../../MODELS/functions/kernels/kern_lq.h"

class DM_Kern_LQ :  public DM_Kern_Base, public Kern_LQ {
    public:
        DM_Kern_LQ();
        DM_Kern_LQ(const Config &c);
};
#endif
