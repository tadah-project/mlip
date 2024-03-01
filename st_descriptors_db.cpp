#include "st_descriptors_db.h"

StDescriptorsDB::StDescriptorsDB(const StructureDB &stdb, Config &config):
    st_descs(stdb.size())
{
    // init all StDescriptors
    for (size_t i=0; i<stdb.size(); ++i)
        st_descs[i]=StDescriptors(stdb(i), config);
}


StDescriptors &StDescriptorsDB::operator()(const size_t s) {
    return st_descs[s];
}
size_t StDescriptorsDB::size() const {
    return st_descs.size();
}
