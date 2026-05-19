#include "tadah/core/element.h"
#include <tadah/mlip/structure_db.h>
#include <tadah/core/periodic_table.h>
#include <cstdio>
#include <cctype>   // For std::isspace
namespace tadah {
namespace mlip {

StructureDB::StructureDB() {
  tadah::core::PeriodicTable::initialize();
}
StructureDB::StructureDB(const tadah::core::Context &context) {
  tadah::core::PeriodicTable::initialize();
  add(context);
}

void StructureDB::add(const std::string fn) {
  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("DBFILE does not exist: "+fn);
  }
  parseFile(fn);
  while (true) {
    structures.push_back(Structure());
    int t = structures.back().read(ifs);

    // did we read structure successfully?
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

    // did we read structure successfully?
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
void StructureDB::add(const StructureDB &stdb) {
  for (const auto &s: stdb) add(s);
}

void StructureDB::remove(size_t i) {
  structures.erase(structures.begin()+i);
}

void StructureDB::add(const tadah::core::Context &context) {
  for (const std::string &s : context("DBFILE")) {
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
std::set<tadah::core::Element> StructureDB::get_unique_elements() const {
  std::set<tadah::core::Element> s;
  for (const auto & st: structures) {
    std::set<tadah::core::Element> u = st.get_unique_elements();
    s.insert(u.cbegin(),u.cend());
  }
  return s;
}
std::set<tadah::core::Element> StructureDB::find_unique_elements(const std::string &fn) {
  std::set<tadah::core::Element> s;
  std::ifstream ifs(fn);
  std::string line;

  if (!ifs.is_open()) {
      core::Logger::instance().log(tadah::core::LogLevel::INFO,
          "StructureDB::find_unique_elements: DBFILE does not exist: " + fn);
    throw std::runtime_error("DBFILE does not exist: "+fn);
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
      s.insert(tadah::core::PeriodicTable::find_by_symbol(symbol));
    }

  }
  return s;
}
std::set<tadah::core::Element> StructureDB::find_unique_elements(const tadah::core::Context &c) {
  std::set<tadah::core::Element> s;
  for (const auto& fn: c("DBFILE")) {
    std::set<tadah::core::Element> temp = find_unique_elements(fn);
    s.insert(temp.begin(), temp.end());
  }
  return s;
}

template <typename T,typename U>                                                   
std::pair<T,U> operator+(const std::pair<T,U>  &l,const std::pair<T,U>  &r) {   
  return {l.first+r.first,l.second+r.second};                                    
}  
std::pair<int,int> StructureDB::count(const tadah::core::Context &context) {
  std::pair<int, int> res=std::make_pair(0,0);  
  for (const std::string &fn : context("DBFILE")) {
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
  for (auto &s : structures) s.clear_nn();
}
void StructureDB::check_atom_key(tadah::core::Context &context, std::set<tadah::core::Element> &unique_elements) {
  bool error=false;

  if (context.contains("ATOM")) {
    // user set this key so here we check does it correspond to unique_elements
    if (unique_elements.size()!=context.size("ATOM")) {
      error=true;
    }

    auto set_it = unique_elements.begin();
    auto atoms_it = context("ATOM").begin();
    while (set_it != unique_elements.end() && atoms_it != context("ATOM").end()) {
      if (set_it->symbol != (*atoms_it))  {
        error=true;
      }
      ++set_it;
      ++atoms_it;
    }
    if (error) {
      throw std::runtime_error("\n"
          "Mismatch between elements in datasets and ATOM in the context file.\n"
          "Please either update the ATOM in the context file or remove ATOM\n"
          "key completely. Tadah! will automatically configure this key.\n"
          );
    }
  } else {
    for (const auto &s : unique_elements) context.add("ATOM", s.symbol);
  }
}
void StructureDB::check_watom_key(tadah::core::Context &context, std::set<tadah::core::Element> &unique_elements) {
  bool error=false;

  if (context.contains("WATOM")) {
    // user set this key so here we check does it correspond to unique_elements
    if (unique_elements.size()!=context.size("WATOM")) {
      error=true;
    }
    if (error) {
      throw std::runtime_error("\n"
          "Mismatch between elements in datasets and WATOM in the context file.\n"
          "Please either update the WATOM in the context file or remove WATOM\n"
          "key completely. In the latter case Tadah! will use default values.\n"
          );
    }
  } else {
    for (const auto &s : unique_elements) context.add("WATOM", s.Z);
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
void StructureDB::dump_to_file(const std::string& filepath, bool append, size_t prec) const
{
    // Determines the file open mode based on append flag
    //   - ios::out means file is opened in output mode
    //   - ios::app means all write operations append at the end
    //   - ios::trunc (the default if append is false) removes existing contents
    std::ios_base::openmode mode = std::ios::out;
    if (append) {
        mode |= std::ios::app;
    } else {
        mode |= std::ios::trunc;
    }

    std::ofstream file(filepath, mode);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return;
    }
    for (const auto &s : structures) {
        s.dump_to_file(file, prec);  
    }
    file.close();
}

std::string StructureDB::summary() const {
  std::string str =  "# of structures : " + std::to_string(structures.size());

  str +=  " | # of atoms : " + std::to_string(calc_natoms());

  str += " | Elements : ";
  std::set<tadah::core::Element> ue = get_unique_elements(); 
  for (const auto &e: ue) str+= e.symbol + " ";
  str+="\n";
  return str;
}
void StructureDB::parseFile(const std::string& filename)
{
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  if (!fin.is_open())
  {
    std::cerr << "Error: could not open file " << filename << "\n";
    return;
  }

  std::size_t header_size = 9;

  // Increase buffer size to speed up I/O on large files.
  static const size_t BUFSIZE = 100ULL << 20; // 100 MiB
  char* buffer = new char[BUFSIZE];
  fin.rdbuf()->pubsetbuf(buffer, BUFSIZE);

  std::vector<size_t> blockLineCounts;
  blockLineCounts.reserve(10000); // Pre-allocate to reduce repeated allocations

  size_t currentBlockCount = 0;
  std::string line;

  while (true)
  {
    if (!std::getline(fin, line))
    {
      // End of file or read error
      break;
    }

    if (isBlankLine(line))
    {
      // We reached the end of the current block
      if (currentBlockCount > 0)
      {
        blockLineCounts.push_back(currentBlockCount-header_size);
        currentBlockCount = 0;
      }
    }
    else
    {
      // Non-empty line => belongs to the current block
      currentBlockCount++;
    }
  }

  // If the last block didn’t end with a blank line, close it out
  if (currentBlockCount > 0)
  {
    blockLineCounts.push_back(currentBlockCount-header_size);
  }

  fin.close();
  delete[] buffer;

  // Print the results
  // std::cout << "Found " << blockLineCounts.size() << " blocks.\n";
  // for (size_t i = 0; i < blockLineCounts.size(); i+=1000)
  // {
  //     std::cout << "Block " << i << " has "
  //               << blockLineCounts[i] << " atoms\n";
  // }
}

bool StructureDB::isBlankLine(const std::string& line) const
{
    for (char c : line)
    {
        if (!std::isspace(static_cast<unsigned char>(c)))
        {
            return false;
        }
    }
    return true;
}

void StructureDB::clear() {
  structures.clear();
  dbidx.clear();
}

} // namespace mlip
} // namespace tadah
