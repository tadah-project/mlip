#ifndef DC_Base_H
#define DC_Base_H

#include "structure.h"
#include "structure_db.h"
#include "st_descriptors.h"
#include "st_descriptors_db.h"

class DC_Base {
    public:
        virtual ~DC_Base() {};
        virtual StDescriptors calc(const Structure &) { return StDescriptors(); };
        virtual StDescriptorsDB calc(const StructureDB &) {return StDescriptorsDB();};
};
#endif
