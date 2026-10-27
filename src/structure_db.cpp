#include "tadah/core/element.h"
#include <tadah/mlip/structure_db.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/utils.h>
#include <tadah/core/logger.h>

#include <map>
#include <stdexcept>
#include <string>
#include <tadah/core/number_format.h>
#include <cstdio>
#include <cctype>   // For std::isspace
namespace tadah {
namespace mlip {

namespace {
/* Reads the next structure of `ifs` into `s`, after any blank lines (one
 * without atoms is a structure): false at the end of the file, where none is
 * left. Errors name the file `fn` and the structure, `position` + 1. */
bool read_next_structure(std::ifstream &ifs, const std::string &fn,
                         size_t position, Structure &s) {
  while (!ifs.eof()) {
    if (ifs.fail())
      throw std::runtime_error("DBFILE " + fn + ": cannot read on after structure " +
                               std::to_string(position));
    int t;
    try {
      t = s.read(ifs);
    } catch (const std::exception &e) {
      throw std::runtime_error("DBFILE " + fn + ": structure " +
                               std::to_string(position + 1) + ": " + e.what());
    }
    if (t == 0) return true;  // t == 1: a blank line
  }
  return false;
}

/* Structure::next_structure for structure `position` + 1 of the file `fn`,
 * whose errors name the file and the structure as read_next_structure's
 * do. */
bool next_structure_of(std::ifstream &ifs, const std::string &fn,
                       size_t position, size_t &natoms) {
  try {
    return Structure::next_structure(ifs, natoms);
  } catch (const std::exception &e) {
    throw std::runtime_error("DBFILE " + fn + ": structure " +
                             std::to_string(position + 1) + ": " + e.what());
  }
}
}  // namespace

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
  size_t read = 0;
  while (true) {
    structures.push_back(Structure());
    int t;
    try {
      t = structures.back().read(ifs);
    } catch (const std::exception &e) {
      structures.pop_back();
      throw std::runtime_error("DBFILE " + fn + ": structure " +
                               std::to_string(read + 1) + ": " + e.what());
    }

    // did we read structure successfully?
    // if not remove last object from the list
    if (t==1) structures.pop_back();
    else ++read;

    if (ifs.eof()) break;
    if (ifs.fail())
      throw std::runtime_error("DBFILE " + fn + ": cannot read on after structure " +
                               std::to_string(read));
  }
  ifs.close();
}
int StructureDB::add(const std::string fn, size_t first, int N) {
  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("DBFILE does not exist: "+fn);
  }

  // move iterator to the first structure we want to read; a file with
  // fewer structures has nothing to load (stop at its end: `first` may be
  // as large as SIZE_MAX). A structure without atoms is skipped as one.
  size_t natoms;
  for (size_t i=0; i<first; ++i) {
    if (!next_structure_of(ifs, fn, i, natoms)) return 0;
  }

  // N structures, blank lines between them not counted.
  size_t read = 0;
  while (N > 0 && read < static_cast<size_t>(N)) {
    Structure s;
    if (!read_next_structure(ifs, fn, first + read, s)) break;
    structures.push_back(std::move(s));
    ++read;
  }
  ifs.close();

  return static_cast<int>(read);
}

int StructureDB::add_selected(const std::string fn,
                              const std::vector<size_t> &indices) {
  if (indices.empty()) return 0;
  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("DBFILE does not exist: "+fn);
  }
  // One pass through the file: each selected structure is read once, in
  // file order, and kept; the others are skipped without being parsed, as
  // count() does. Memory grows with the number of selected structures.
  std::map<size_t, Structure> selected;
  for (const size_t i : indices) selected.emplace(i, Structure());
  const size_t last = selected.rbegin()->first;
  auto past_end = [&](size_t in_file) {
    return std::runtime_error("DBFILE " + fn + ": structure " +
                              std::to_string(last + 1) +
                              " is past the end: the file has " +
                              std::to_string(in_file) + " structures");
  };
  size_t position = 0;
  for (auto &entry : selected) {
    size_t natoms;
    for (; position < entry.first; ++position)
      if (!next_structure_of(ifs, fn, position, natoms)) throw past_end(position);
    if (!read_next_structure(ifs, fn, position, entry.second))
      throw past_end(position);
    ++position;
  }
  for (const size_t i : indices) add(selected.at(i));
  return static_cast<int>(indices.size());
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
    throw std::runtime_error("DBFILE does not exist: "+fn);
  }

  size_t position = 0;  // the structures read so far
  while (std::getline(ifs, line)) {
    // Blank lines before a label are skipped (a line of only whitespace,
    // '\r' included, is blank, as in Structure::read).
    if (tadah::core::is_blank_line(line)) continue;
    const std::string label = line;
    ++position;
    // the second line could be energy or
    // a scalling factors eweight fweight sweight
    std::getline(ifs,line);  
    std::stringstream stream(line);
    size_t count = std::distance(std::istream_iterator<std::string>(stream),
        std::istream_iterator<std::string>());

    if (count == 3)
      std::getline(ifs,line);

    // The cell and stress lines, as Structure::read reads them.
    try {
      (void)detail::read_cell_stress_lines(ifs);
    } catch (const std::exception &e) {
      throw std::runtime_error("DBFILE " + fn + ": structure " +
                               std::to_string(position) + ": structure '" +
                               label + "': " + e.what());
    }

    while (std::getline(ifs, line)) {
      if (tadah::core::is_blank_line(line)) break;
      std::istringstream atom(line);
      std::string symbol;
      atom >> symbol;
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
  // A structure without atoms counts as one (StructureDB::add(file) reads
  // it); only the end of the file ends the count.
  size_t natoms;
  while (next_structure_of(ifs, fn, static_cast<size_t>(nstruc), natoms)) {
    natoms_tot += static_cast<int>(natoms);
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
      std::string named, have;
      for (const auto &a : context("ATOM"))
        named += (named.empty() ? "" : " ") + a;
      for (const auto &e : unique_elements)
        have += (have.empty() ? "" : " ") + e.symbol;
      throw std::runtime_error(
          "the ATOM line of the training configuration names " + named +
          ", and the datasets have " + have + "; an ATOM line that lists "
          "these elements in this order, or none, is accepted");
    }
  } else {
    for (const auto &s : unique_elements) context.add("ATOM", s.symbol);
  }
}
void StructureDB::check_watom_key(tadah::core::Context &context, std::set<tadah::core::Element> &unique_elements) {
  if (context.contains("WATOM")) {
    const size_t n = context.size("WATOM");
    if (n == 0) {
      throw std::runtime_error("\nWATOM is empty.\n");
    }
    const bool per_species =
      tadah::core::is_number(context.get<std::string>("WATOM", 0));
    if (per_species) {
      if (unique_elements.size() != n) {
        std::string have;
        for (const auto &e : unique_elements)
          have += (have.empty() ? "" : " ") + e.symbol;
        throw std::runtime_error(
            "WATOM of the training configuration has " + std::to_string(n) +
            (n == 1 ? " value" : " values") + " for the " +
            std::to_string(unique_elements.size()) + " elements of the "
            "datasets (" + have + "); one WATOM value for each element of "
            "the datasets, or no WATOM line, is accepted");
      }
    } else {
      if (n % 3 != 0) {
        throw std::runtime_error(
            "WATOM in the pair form takes '<symbol> <symbol> <value>' "
            "triples, and the line has " + std::to_string(n) + " entries");
      }
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
    if (!file.is_open())
        throw std::runtime_error("Could not open file for writing: " + filepath);
    for (const auto &s : structures) {
        s.dump_to_file(file, prec);  
    }
    file.close();
    if (file.fail())
        throw std::runtime_error("Could not write the file: " + filepath);
}

namespace {
// NUMERIC for a written structure file, looked up without default
// insertion; an explicit value below 17 is honoured with a WARNING.
size_t structure_file_digits(const tadah::core::Context &ctx,
                             const std::string &numbers,
                             const std::string &consequence) {
  const tadah::core::PotDigits pd = tadah::core::resolve_pot_digits(ctx);
  if (pd.explicit_ && pd.digits < tadah::core::kRoundTripDigits) {
    const std::string n = std::to_string(pd.digits);
    tadah::core::Logger::instance().log(tadah::core::LogLevel::WARNING,
        "NUMERIC " + n + " limits " + numbers + " to " + n +
        " significant digits: values do not round-trip exactly "
        "(a double needs " + std::to_string(tadah::core::kRoundTripDigits) +
        "), so " + consequence + ". Omit NUMERIC (default " +
        std::to_string(tadah::core::kPotDigitsDefault) +
        ", exact) or set NUMERIC >= " +
        std::to_string(tadah::core::kRoundTripDigits) + ".");
  }
  return static_cast<size_t>(pd.digits);
}
}  // namespace

size_t StructureDB::dataset_digits(const tadah::core::Context &ctx) {
  return structure_file_digits(
      ctx, "every number in the written dataset file",
      "a dataset reloaded from it differs from the one in memory");
}

size_t StructureDB::export_digits(const tadah::core::Context &ctx,
                                  const std::string &file_noun) {
  return structure_file_digits(
      ctx, "every real number in the written " + file_noun,
      "the exported structure differs from the one in memory");
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
