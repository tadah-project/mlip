#include <tadah/mlip/structure_db.h>
#include <tadah/core/periodic_table.h>
#include <cstdio>

StructureDB::StructureDB() {
  PeriodicTable::initialize();
}
StructureDB::StructureDB(Config &config) {
  PeriodicTable::initialize();
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
int StructureDB::add(const std::string fn, size_t first, int N) {
  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("DBFILE does not exist: "+fn);
  }

  // move iterator to the first structure we want to read
  for (size_t i=0; i<first; ++i) {
    Structure::next_structure(ifs);
  }

  int count=0;
  while (count++<N) {
    structures.push_back(Structure());
    int t = structures.back().read(ifs);

    // did we read structure succesfully?
    // if not remove last object from the list
    if (t==1) structures.pop_back();

    if (ifs.eof()) break;
  }
  ifs.close();

  return --count;
}

void StructureDB::add(const Structure &s) {
  structures.push_back(s);
}

void StructureDB::remove(size_t i) {
  structures.erase(structures.begin()+i);
}

void StructureDB::add(Config &config) {
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
std::set<Element> StructureDB::get_unique_elements() const {
  std::set<Element> s;
  for (const auto & st: structures) s.insert(
      st.unique_elements.begin(),st.unique_elements.end());
  return s;
}
std::set<Element> StructureDB::find_unique_elements(const std::string &fn) {
  std::set<Element> s;
  std::ifstream ifs(fn);
  std::string line;

  if (!ifs.is_open()) {
    std::cerr << "Could not open the file." << std::endl;
  }

  char symbol[3];
  while (std::getline(ifs, line)) {
    // the second line could be energy or
    // a scalling factors eweight fweight sweight
    std::getline(ifs,line);  
    std::stringstream stream(line);
    size_t count = std::distance(std::istream_iterator<std::string>(stream),
        std::istream_iterator<std::string>());

    if (count == 3)
      std::getline(ifs,line);

    for (size_t i=0; i<6; ++i)
      std::getline(ifs,line);

    while (std::getline(ifs, line)) {
      if(line.empty()) break;
      sscanf(line.c_str(), "%2s", symbol);
      s.insert(PeriodicTable::find_by_symbol(symbol));
    }

  }
  return s;
}
std::set<Element> StructureDB::find_unique_elements(const Config &c) {
  std::set<Element> s;
  for (const auto& fn: c("DBFILE")) {
    std::set<Element> temp = find_unique_elements(fn);
    s.insert(temp.begin(), temp.end());
  }
  return s;
}

template <typename T,typename U>                                                   
std::pair<T,U> operator+(const std::pair<T,U>  &l,const std::pair<T,U>  &r) {   
  return {l.first+r.first,l.second+r.second};                                    
}  
std::pair<int,int> StructureDB::count(const Config &config) {
  std::pair<int, int> res=std::make_pair(0,0);  
  for (const std::string &fn : config("DBFILE")) {
    res = res + count(fn);
  }
  return res;
}

std::pair<int,int> StructureDB::count(const std::string fn){
  int nstruc=0;
  int natoms_tot=0;

  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("DBFILE does not exist: "+fn);
  }
  int natoms = Structure::next_structure(ifs);
  while(natoms) {
    natoms_tot += natoms;
    natoms = Structure::next_structure(ifs);
    nstruc++;
  }

  std::pair<int,int> res = std::make_pair(nstruc,natoms_tot);
  return res;
}
void StructureDB::clear_nn() {
  for (auto &struc: structures) struc.clear_nn();
}
void StructureDB::check_atoms_key(Config &config, std::set<Element> &unique_elements) {
  bool error=false;

  if (config.exist("ATOMS")) {
    // user set this key so here we check does it correspond to unique_elements
    if (unique_elements.size()!=config.size("ATOMS")) {
      error=true;
    }

    auto set_it = unique_elements.begin();
    auto atoms_it = config("ATOMS").begin();
    while (set_it != unique_elements.end() && atoms_it != config("ATOMS").end()) {
      if (set_it->symbol != (*atoms_it))  {
        error=true;
      }
      ++set_it;
      ++atoms_it;
    }
    if (error) {
      throw std::runtime_error("\n"
          "Mismatch between elements in datasets and ATOMS in the config file.\n"
          "Please either update the ATOMS in the config file or remove ATOMS\n"
          "key completely. Tadah! will automatically configure this key.\n"
          );
    }
  } else {
    for (const auto &s : unique_elements) config.add("ATOMS", s.symbol);
  }
}
void StructureDB::check_watoms_key(Config &config, std::set<Element> &unique_elements) {
  bool error=false;

  if (config.exist("WATOMS")) {
    // user set this key so here we check does it correspond to unique_elements
    if (unique_elements.size()!=config.size("WATOMS")) {
      error=true;
    }
    if (error) {
      throw std::runtime_error("\n"
          "Mismatch between elements in datasets and WATOMS in the config file.\n"
          "Please either update the WATOMS in the config file or remove WATOMS\n"
          "key completely. In the latter case Tadah! will use default values.\n"
          );
    }
  } else {
    for (const auto &s : unique_elements) config.add("WATOMS", s.Z);
  }
}

std::vector<Structure>::iterator StructureDB::begin() { 
    return structures.begin(); 
}

std::vector<Structure>::iterator StructureDB::end() { 
    return structures.end(); 
}

std::vector<Structure>::const_iterator StructureDB::begin() const { 
    return structures.cbegin(); 
}

std::vector<Structure>::const_iterator StructureDB::end() const { 
    return structures.cend(); 
}
