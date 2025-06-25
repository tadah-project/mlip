#pragma   once
#include <tadah/mlip/structure.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/m_predict.h>

namespace tadah::mlip
{

/*  Stateless functor – may be instantiated once and reused.
 *  Thread-safe, needs OpenMP.                                      */
class FastEnergyEvaluator
{
public:
    /*  Compute energy for ONE Structure.  The DC_Selector must be
     *  fully initialised beforehand so that d2b/dmb/fcut pointers
     *  are valid and descriptor offsets are set.                   */
    double operator()(const Structure&            st,
                      const core::Context&        ctx,
                      const models::DC_Selector&  dc,
                      const models::M_Predict&    model) const;
};

}   // namespace tadah::mlip
