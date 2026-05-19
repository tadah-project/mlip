#include <tadah/mlip/structure.h>
#include <tadah/core/periodic_table.h>

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <sstream>
#include <algorithm>
#include <cstddef>
namespace tadah {
namespace mlip {

/**
 * Creates and initializes a Structure object with default parameters.
 * Reason: Ensures all member variables are set to meaningful defaults.
 *         tadah::core::PeriodicTable is initialized to allow symbol lookups.
 *
 * Example usage:
 *     Structure st; // Creates an empty structure with zeroed data
 */
Structure::Structure() {
  // tadah::core::PeriodicTable is prepared for atomic symbol lookups
  tadah::core::PeriodicTable::initialize();

  // Zero or default values assigned to all members so there are no uninitialized values
  label.clear();
  energy    = 0.0;
  eweight   = 1.0;
  fweight   = 1.0;
  sweight   = 1.0;
  T         = 0.0;

  cell.set_zero();
  stress.set_zero();

  atoms.clear();
}

/**
 * Returns a const reference to the i-th Atom in the atoms container.
 * Reason: Provides read-only access to the atom at index i.
 */
const Atom& Structure::operator()(const size_t i) const {
  return atoms[i];
}

/**
 * Adds a new Atom to the atoms container.
 * Reason: Allows extension of the structure with additional atoms.
 *
 * Example usage:
 *     Structure st;
 *     Atom a(tadah::core::Element("H"), 0.0, 0.1, 0.2);
 *     st.add_atom(a);
 */
void Structure::add_atom(Atom a) {
  atoms.push_back(a);
}

/**
 * Removes the i-th Atom from the atoms container.
 * Reason: Allows selective removal of atoms from the structure.
 *
 * Example usage:
 *     st.remove_atom(0); // Removes the first atom
 */
void Structure::remove_atom(const size_t i) {
  atoms.erase(atoms.begin() + i);
}

/**
 * Returns the position of the n-th neighbor of atom i, taking into account periodic images.
 * Reason: Shifts the neighbor's position by the periodic cell vector to emulate periodic boundaries.
 *
 * Example usage:
 *     // Retrieve the neighbor's position (the 2nd neighbor of atom 0)
 *     tadah::core::Vec3d neighborPos = st.nn_pos(0, 1);
 */
// tadah::core::Vec3d Structure::nn_pos(const size_t i, const size_t n) const {
//   // (A) Global index of this neighbor
//   const size_t neighborIndex = nn->idx[i][n];
//
//   // (B) Atom's original "unshifted" position
//   const tadah::core::Vec3d &posNeighbor = atoms[neighborIndex].position;
//
//   // (C) Convert the stored shift -> real displacement shiftDisp
//   //     If nn->shift[i][n] is an integer triple (n1, n2, n3),
//   //     multiply by the cell. Otherwise, if it's already in real space, it may simply be added.
//   tadah::core::Vec3d shift = nn->shift[i][n]; // Possibly an integer triple
//   tadah::core::Vec3d shiftDisp;
//   shiftDisp[0] = shift[0] * cell(0,0) + shift[1] * cell(0,1) + shift[2] * cell(0,2);
//   shiftDisp[1] = shift[0] * cell(1,0) + shift[1] * cell(1,1) + shift[2] * cell(1,2);
//   shiftDisp[2] = shift[0] * cell(2,0) + shift[1] * cell(2,1) + shift[2] * cell(2,2);
//   // shiftDisp[0] = shift[0] * cell(0,0) + shift[1] * cell(1,0) + shift[2] * cell(2,0);
//   // shiftDisp[1] = shift[0] * cell(0,1) + shift[1] * cell(1,1) + shift[2] * cell(2,1);
//   // shiftDisp[2] = shift[0] * cell(0,2) + shift[1] * cell(1,2) + shift[2] * cell(2,2);
//
//   // (D) Final neighbor position = unshifted position + shiftDisp
//   return posNeighbor + shiftDisp;
// }
tadah::core::Vec3d Structure::nn_pos(const size_t i, const size_t n) const 
{
  // (A) index of neighbor
  const size_t neighborIndex    = nn->idx  [i][n];

  // (B) that neighbor's base ("home cell") position
  const tadah::core::Vec3d &posNeighbor = atoms[neighborIndex].position;

  // (C) Convert the stored shift -> real displacement
  //     We said nn->shift[i][n] is the integer triple (n1,n2,n3).
  const auto& shift = nn->shift[i][n];

  // row-based multiplication
  tadah::core::Vec3d shiftDisp;
  shiftDisp[0] = shift[0]*cell(0,0) + shift[1]*cell(1,0) + shift[2]*cell(2,0);
  shiftDisp[1] = shift[0]*cell(0,1) + shift[1]*cell(1,1) + shift[2]*cell(2,1);
  shiftDisp[2] = shift[0]*cell(0,2) + shift[1]*cell(1,2) + shift[2]*cell(2,2);

  // (D) final position
  return posNeighbor + shiftDisp;
}


/**
 * Returns the number of atoms contained in the atoms vector.
 * Reason: Convenient way to obtain the size of the system.
 */
size_t Structure::natoms() const {
  return atoms.size();
}

std::unordered_map<int,size_t> Structure::get_natoms_map() const {
  std::set<core::Element> elements = get_unique_elements();
  std::unordered_map<int,size_t> natoms_map;
  for (const auto &element : elements) {
    natoms_map[element.Z] = 0;
  }
  for (const auto &atom : atoms) {
    natoms_map[atom.Z]++;
  }
  return natoms_map;
}

/**
 * Returns a number of nearest neighbors for the i-th Atom.
 * Reason: Gives fast access to the local environment size of a specific atom.
 */
size_t Structure::nn_size(size_t i) const {
  return nn ? nn->nn_size(i) : 0;
}

/**
 * Reads structure data from the given file stream. 
 * Reason: Populates members of Structure from a file input.
 * 
 * Returns 0 on success, 1 on an empty or invalid line.
 *
 * Example usage:
 *     std::ifstream ifs("my_structure.dat");
 *     Structure st;
 *     int status = st.read(ifs);
 */
int Structure::read(std::ifstream &ifs) {
  std::string line;
  // Ignore extra empty lines
  std::getline(ifs, line);
  if(line.empty()) return 1;

  // The first line read is used as a label
  label = line;

  // The second line might be energy, or scaling factors eweight fweight sweight, or energy T
  std::getline(ifs, line);
  std::stringstream stream(line);
  std::string temp;
  size_t count = 0;

  while(stream >> temp) { ++count; }
  stream.clear();
  stream.seekg(0, std::ios::beg);

  try {
  if (count == 3) {
    // eweight, fweight, sweight
    stream >> eweight >> fweight >> sweight;
    // read the next line for energy and T
    std::getline(ifs, line);
    std::stringstream stream2(line);
    stream2 >> energy;
    if(stream2) stream2 >> T;
  }
  else if (count == 2) {
    // energy, T
    stream >> energy >> T;
  }
  else {
    // only energy
    energy = std::stod(line);
  }

  // 3x3 cell matrix
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ifs >> cell(i, j);

  // 3x3 stress matrix
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ifs >> stress(i, j);

  // Move to next line to start reading atoms
  std::getline(ifs, line);

  // Clear existing atoms
  atoms.clear();

  // Local variables for reading each line
  char symbol[3];
  double px, py, pz, fx, fy, fz;

  // Read in each atom
  while(std::getline(ifs, line)) {
    if(line.empty())  break;
    if(line == "\r")  break; // Windows newline detection
    
    std::istringstream tmp(line);
    tmp >> symbol >> px >> py >> pz >> fx >> fy >> fz;
    tadah::core::Element e = tadah::core::PeriodicTable::find_by_symbol(symbol);
    atoms.push_back(Atom(e, px, py, pz, fx, fy, fz));
  }
  } catch (const std::exception &e) {
    std::cerr << "Error reading structure: " << e.what() << std::endl;
    std::cerr << "The error occurred while reading line: " << line << std::endl;
    throw std::runtime_error("Error reading structure from file.");
  }
  return 0;
}

/**
 * Reads the first structure available in a file.
 * Reason: Simplifies workflow by only reading one structure from the provided file path.
 *
 * Throws std::runtime_error if the file cannot be opened.
 *
 * Example usage:
 *     Structure st;
 *     st.read("structure_input.dat");
 */
void Structure::read(const std::string fn) {
  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("File does not exist: " + fn);
  }
  read(ifs);
  ifs.close();
}

/**
 * Saves the current structure data to a given output file stream.
 * Reason: Writes out all attributes of Structure in a formatted manner.
 *
 * Example usage:
 *     std::ofstream ofs("my_structure_output.dat");
 *     st.save(ofs);
 */
void Structure::save(std::ofstream &ofs) const {
  ofs << label << std::endl;
  ofs << eweight << " " << fweight << " " << sweight << std::endl;
  ofs << energy << std::endl;
  for (int i = 0; i < 3; ++i) {
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << cell(i,0);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << cell(i,1);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << cell(i,2);
    ofs << std::endl;
  }

  for (int i = 0; i < 3; ++i) {
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << stress(i,0);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << stress(i,1);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << stress(i,2);
    ofs << std::endl;
  }

  for (const Atom &a : atoms) {
    ofs << a.symbol[0] << a.symbol[1];
    ofs << std::setw(w-2) << std::fixed << std::right
        << std::setprecision(p) << a.position(0);
    ofs << std::setw(w) << std::fixed << std::right
        << std::setprecision(p) << a.position(1);
    ofs << std::setw(w) << std::fixed << std::right
        << std::setprecision(p) << a.position(2);
    ofs << std::setw(w) << std::fixed << std::right
        << std::setprecision(p) << a.force(0);
    ofs << std::setw(w) << std::fixed << std::right
        << std::setprecision(p) << a.force(1);
    ofs << std::setw(w) << std::fixed << std::right
        << std::setprecision(p) << a.force(1); // Notice potential mistake: repeated a.force(1)
    ofs << std::endl;
  }
}

/**
 * Saves the current structure data to a file.
 * Reason: Directly writes all attributes of Structure to a user-specified file path.
 *
 * Example usage:
 *     st.save("my_structure_output.dat");
 *
 * Warning: Overwrites the file if it already exists.
 */
void Structure::save(const std::string fn) const {
  std::ofstream ofs(fn);
  save(ofs);
  ofs.close();
}

/**
 * Returns the corresponding i-index (ii) in the neighbor list for atom j, that matches i-j.
 * Reason: Needed to fully specify i-j interactions for many-body force calculations,
 *         especially if rcut > half of box length (duplicated interactions).
 *
 * Example usage:
 *     size_t ii = get_nn_iindex(i, j, jj);
 */
size_t Structure::get_nn_iindex(const size_t i, const size_t j, const size_t jj) const {
  tadah::core::Vec3d shift_ijj = -nn->shift[i][jj];
  size_t ii = 0;
  for (size_t x=0; x<nn->idx[j].size(); ++x)
  {
    if (nn->idx[j][x]==i && nn->shift[j][x]==shift_ijj) break;
    ii++;
  }
  return ii;
}

/**
 * Returns the volume of the structure by calculating det(cell).
 * Reason: Volume is essential for many physical calculations (density, pressure, etc.).
 */
double Structure::get_volume() const {
  return cell.row(0)*(cell.row(1).cross(cell.row(2)));
}

/**
 * Returns the density of the structure in g/cm^3.
 * Reason: A common physical property used in various analyses.
 *
 * Example usage:
 *     double density = st.get_density();
 */
double Structure::get_density() const {
  double V = get_volume();
  V *= 1e-24; // Converts Angstrom^3 to cm^3
  double amu = 1.66053906660e-24; // 1 atomic mass unit in grams
  double mass = 0.0;
  for (const auto& a : atoms) {
    mass += tadah::core::PeriodicTable::get_mass(a.Z); // sum of atomic masses
  }
  return amu * mass / V;
}

/**
 * Returns the temperature, T, of the structure in K.
 * Reason: Provides quick inspection or usage in further calculations.
 */
double Structure::get_temperature() const {
  return T;
}

/**
 * Returns the virial pressure, units of (energy / distance^3).
 * Reason: The virial pressure is needed for dealing with stress or internal equations of state.
 */
double Structure::get_virial_pressure() const {
  return stress.trace() / get_volume() / 3.0;
}

/**
 * Returns the total pressure at the given temperature T with optional Boltzmann constant (kB).
 * Reason: Includes ideal-gas-like component from T plus the virial contribution from the stress.
 *
 * Example usage:
 *     double P = st.get_pressure(300.0); // for T = 300 K
 */
double Structure::get_pressure(double T, double kB) const {
  double vpress = get_virial_pressure();
  return vpress + natoms() * kB * T / get_volume();
}

/**
 * Equality operator compares cell, stress, number of atoms, weights, energy, 
 * and all positions/forces to check if two structures are identical.
 * Reason: Facilitates quick checks to confirm if two structures match exactly.
 */
bool Structure::operator==(const Structure& st) const {
  double EPSILON = std::numeric_limits<double>::epsilon();

  bool result =
    cell.isApprox(st.cell, EPSILON) &&
    stress.isApprox(st.stress, EPSILON) &&
    (natoms() == st.natoms()) &&
    (std::fabs(eweight - st.eweight) < EPSILON) &&
    (std::fabs(fweight - st.fweight) < EPSILON) &&
    (std::fabs(sweight - st.sweight) < EPSILON) &&
    (std::fabs(energy - st.energy) < EPSILON);

  if (!result) return false;

  for (size_t i = 0; i < natoms(); ++i) {
    // Compares Atom-level data
    result = (atoms[i] == st.atoms[i]);
    if (!result) return false;
  }
  return true;
}

/**
 * Checks if two structures match in cell vectors, number of atoms, species, and coordinates,
 * ignoring order of atoms and not comparing energies, forces, or labels.
 * Reason: More lenient check to see if structures represent the "same" geometry.
 */
bool Structure::is_the_same(const Structure& st, double thr) const {
  bool result =
    cell.isApprox(st.cell, thr) &&
    (natoms() == st.natoms());

  if (!result) return false;

  size_t count = 0;
  for (size_t i = 0; i < natoms(); ++i) {
    for (size_t j = i; j < natoms(); ++j) {
      if (atoms[i].is_the_same(st.atoms[j], thr)) {
        count++;
      }
    }
  }
  return (count == natoms());
}

/**
 * Moves the file stream to skip over the next structure. 
 * Reason: Used to step through multiple structures in a single file.
 *
 * Returns the number of atoms of the skipped structure or 0 if there are no more.
 */
int Structure::next_structure(std::ifstream &ifs) {
  std::string line;
  // Skip the structure label line
  std::getline(ifs, line);
  if(line.empty()) return 0;

  // Read the second line
  std::getline(ifs, line);

  // Check for possible eweight/fweight/sweight line
  std::stringstream stream(line);
  std::string temp;
  size_t count = 0;
  while(stream >> temp) {
    ++count;
  }
  // If second line has 3 entries, skip another line 
  if (count == 3)
    std::getline(ifs, line);

  // Skip cell and stress lines (3+3 = 6 lines)
  for (size_t i = 0; i < 6; ++i)
    std::getline(ifs, line);

  int natoms = 0;
  while(std::getline(ifs, line)) {
    if (line.empty()) break;
    if (line == "\r") break;
    natoms++;
  }
  return natoms;
}

void Structure::clear_nn() {
  nn.reset();
}

/**
 * Returns an iterator to the first Atom in the atoms vector.
 * Reason: Enables range-based for loops: for (auto& a : st).
 */
std::vector<Atom>::iterator Structure::begin() {
  return atoms.begin();
}

/**
 * Returns an iterator to "one-past-the-end" Atom in the atoms vector.
 * Reason: Enables range-based for loops in conjunction with begin().
 */
std::vector<Atom>::iterator Structure::end() {
  return atoms.end();
}

/**
 * Returns a const_iterator to the first Atom in the atoms vector.
 * Reason: Const-correctness for read-only range-based for loops.
 */
std::vector<Atom>::const_iterator Structure::begin() const {
  return atoms.cbegin();
}

/**
 * Returns a const_iterator to "one-past-the-end" Atom in the atoms vector.
 * Reason: Const-correctness for read-only range-based for loops.
 */
std::vector<Atom>::const_iterator Structure::end() const {
  return atoms.cend();
}

/**
 * Dumps the content of the structure to a stream in a detailed format.
 * Reason: Provides a user-friendly ASCII representation of the Structure.
 *
 * Example usage:
 *     st.dump_to_file(std::cout);
 */
void Structure::dump_to_file(std::ostream &file, size_t prec) const {
  const int n = 5;
  file << label << std::endl;
  file << std::fixed << std::setprecision(prec);
  file << eweight << " " << fweight << " " << sweight << std::endl;
  file << energy << " " << T << std::endl;

  file
    << std::setw(prec + n) << cell(0,0) << " "
    << std::setw(prec + n) << cell(0,1) << " "
    << std::setw(prec + n) << cell(0,2) << " " << std::endl
    << std::setw(prec + n) << cell(1,0) << " "
    << std::setw(prec + n) << cell(1,1) << " "
    << std::setw(prec + n) << cell(1,2) << " " << std::endl
    << std::setw(prec + n) << cell(2,0) << " "
    << std::setw(prec + n) << cell(2,1) << " "
    << std::setw(prec + n) << cell(2,2) << " " << std::endl;

  file
    << std::setw(prec + n) << stress(0,0) << " "
    << std::setw(prec + n) << stress(0,1) << " "
    << std::setw(prec + n) << stress(0,2) << " " << std::endl
    << std::setw(prec + n) << stress(1,0) << " "
    << std::setw(prec + n) << stress(1,1) << " "
    << std::setw(prec + n) << stress(1,2) << " " << std::endl
    << std::setw(prec + n) << stress(2,0) << " "
    << std::setw(prec + n) << stress(2,1) << " "
    << std::setw(prec + n) << stress(2,2) << " " << std::endl;

  for (const auto &a : atoms) {
    file << std::setw(2) << a.symbol << " "
         << std::setw(prec + n) << a.position[0] << " "
         << std::setw(prec + n) << a.position[1] << " "
         << std::setw(prec + n) << a.position[2] << " "
         << std::setw(prec + n) << a.force[0] << " "
         << std::setw(prec + n) << a.force[1] << " "
         << std::setw(prec + n) << a.force[2] << std::endl;
  }
  file << std::endl;
}

/**
 * Dumps the content of the structure to a file in a detailed format.
 * Reason: Allows archival of structure data, typically appended. 
 *
 * Example usage:
 *     st.dump_to_file("backup_structure.dat");
 */
void Structure::dump_to_file(const std::string &filepath, size_t prec) const {
  std::ofstream file(filepath, std::ios::app); // Open in append mode
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
    return;
  }
  dump_to_file(file, prec);
  file.close();
}

/**
 * Returns a set of unique chemical elements in the structure.
 * Reason: Identifies the chemical variety present in the structure.
 *
 * Example usage:
 *     std::set<tadah::core::Element> elements = st.get_unique_elements();
 */
std::set<tadah::core::Element> Structure::get_unique_elements() const {
  std::set<tadah::core::Element> unique_elements;
  for (const auto &a : atoms) {
    unique_elements.insert(a);
  }
  return unique_elements;
}

} // namespace mlip
} // namespace tadah
