#ifndef DC_Base_H
#define DC_Base_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/st_descriptors.h>
#include <tadah/mlip/st_descriptors_db.h>
namespace tadah {
namespace mlip {

class DC_Base {
    public:
        virtual ~DC_Base() {};
        virtual StDescriptors calc(const Structure &) const { return StDescriptors(); };
        virtual StDescriptorsDB calc(const StructureDB &) const {return StDescriptorsDB();};
};
} // namespace mlip
} // namespace tadah
#endif
