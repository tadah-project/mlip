#include "structure_db.h"

StructureDB::StructureDB() {}
StructureDB::StructureDB(const Config &config) {
    add(config);
}

void StructureDB::add(const std::string fn) {
    std::ifstream ifs(fn);
    if (!ifs.is_open()) {
        throw std::runtime_error("DBFILE does not exist: "+fn);
    }
    while (true) {
        structures.push_back(Structure());
        int t = structures.back().read(ifs);

        // did we read structure succesfully?
        // if not remove last object from the list
        if (t==1) structures.pop_back();

        if (ifs.eof()) break;
    }
    ifs.close();
}

void StructureDB::add(const Structure &s) {
    structures.push_back(s);
}

void StructureDB::remove(size_t i) {
    structures.erase(structures.begin()+i);
}

void StructureDB::add(const Config &config) {
    for (const std::string &s : config("DBFILE")) {
        dbidx.push_back(size());
        add(s);
    }
    dbidx.push_back(size());
}

size_t StructureDB::size() const {
    return structures.size();
}

size_t StructureDB::size(size_t n) const {
    return dbidx[n+1]-dbidx[n];
}

Structure &StructureDB::operator()(size_t s) {
    return structures[s];
}

const Structure &StructureDB::operator()(size_t s) const {
    return structures[s];
}

Atom &StructureDB::operator()(size_t s, size_t a) {
    return structures[s].atoms[a];
}
size_t StructureDB::calc_natoms() const {
    size_t natoms=0;
    for (auto struc: structures) natoms += struc.natoms();
    return natoms;
}
size_t StructureDB::calc_natoms(size_t n) const {
    size_t start = dbidx[n];
    size_t stop = dbidx[n+1];
    size_t natoms=0;
    for (size_t i=start; i<stop; ++i) {
        natoms += (*this)(i).natoms();
    }
    return natoms;
}
std::unordered_set<Element> StructureDB::get_unique_elements() {
    std::unordered_set<Element> s;
    for (const auto & st: structures) s.insert(
            st.unique_elements.begin(),st.unique_elements.end());
    return s;
}
