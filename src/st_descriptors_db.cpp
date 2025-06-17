#include <tadah/mlip/st_descriptors_db.h>
namespace tadah {
namespace mlip {

StDescriptorsDB::StDescriptorsDB(const StructureDB &stdb, tadah::core::Context &context):
    st_descs(stdb.size())
{
    // init all StDescriptors
    for (size_t i=0; i<stdb.size(); ++i)
        st_descs[i]=StDescriptors(stdb(i), context);
}


StDescriptors &StDescriptorsDB::operator()(const size_t s) {
    return st_descs[s];
}
size_t StDescriptorsDB::size() const {
    return st_descs.size();
}
void StDescriptorsDB::add(const StDescriptors &st_d) {
  st_descs.push_back(st_d);
}
std::vector<StDescriptors>::iterator StDescriptorsDB::begin() { 
    return st_descs.begin(); 
}

std::vector<StDescriptors>::iterator StDescriptorsDB::end() { 
    return st_descs.end(); 
}

std::vector<StDescriptors>::const_iterator StDescriptorsDB::begin() const { 
    return st_descs.cbegin(); 
}

std::vector<StDescriptors>::const_iterator StDescriptorsDB::end() const { 
    return st_descs.cend(); 
}
} // namespace mlip
} // namespace tadah
