#ifndef DC_Base_H
#define DC_Base_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/st_descriptors_db.h>
#include <tadah/mlip/design_matrix/column_block.h>
#include <tadah/core/context.h>
namespace tadah {
namespace mlip {

class DC_Base {
    public:
        virtual ~DC_Base() {};
        virtual StDescriptors calc(const Structure &) const { return StDescriptors(); };
        virtual StDescriptorsDB calc(const StructureDB &) const {return StDescriptorsDB();};
        /** \brief Per-block column layout for the design matrix.
         *  Default returns an empty layout — concrete DescriptorsCalc
         *  overrides this. The DesignMatrix audit (Phase 2) skips its
         *  validate() check when the layout is empty, so legacy
         *  DC_Base-only consumers stay byte-identical to before.
         */
        virtual ColumnLayout layout(const tadah::core::Context &) const {
          return ColumnLayout{};
        }
};
} // namespace mlip
} // namespace tadah
#endif
