#include <tadah/mlip/structure.h>

#include <array>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/utils.h>

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <sstream>
#include <algorithm>
#include <cstddef>
#include <string_view>
#include <vector>
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
tadah::core::Vec3d Structure::nn_pos(const size_t i, const size_t n) const
{
  // Real-space displacement is precomputed at NN-build time
  // (NNFinder::populate_disp). The old form computed
  // shift[0]*cell.row(0) + shift[1]*cell.row(1) + shift[2]*cell.row(2)
  // on every call which dominated descriptor-loop inner cost (~12% of
  // total CPU in perf). The disp cache makes nn_pos a pure vector add.
  const tadah::core::Vec3d &posNeighbor = atoms[nn->idx[i][n]].position;
  const tadah::core::Vec3d &shiftDisp   = nn->disp[i][n];
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
namespace {
/* The whole token `tok` as a finite number; throws std::runtime_error
 * naming `what` and the token otherwise ('nan', 'inf', '1e309', '1.0x'). */
double finite_value(const std::string &tok, const std::string &what) {
  double v = 0.0;
  if (!tadah::core::parse_number(tok, v) || !std::isfinite(v))
    throw std::runtime_error(what + " '" + tok + "' is not a finite number");
  return v;
}

/* finite_value() of a priority (eweight, fweight, sweight), which must
 * also be >= 0: 0 leaves the structure's energy, forces or stress out of a
 * fit, and a negative priority is none. */
double weight_value(const std::string &tok, const std::string &what) {
  const double v = finite_value(tok, what);
  if (v < 0.0)
    throw std::runtime_error(what + " '" + tok + "' is negative: a priority "
                             "is a number >= 0 (0 leaves the structure's " +
                             (what == "eweight"   ? "energy"
                              : what == "fweight" ? "forces"
                                                  : "stress") +
                             " out of a fit)");
  return v;
}

std::vector<std::string> tokens_of(const std::string &line) {
  std::vector<std::string> t;
  std::istringstream in(line);
  for (std::string w; in >> w;) t.push_back(w);
  return t;
}
}  // namespace

namespace detail {

std::array<std::string, 18> read_cell_stress_lines(std::istream &in) {
  std::array<std::string, 18> fields;
  std::string line;
  for (int b = 0; b < 2; ++b) {
    const char *const block = b == 0 ? "cell" : "stress";
    for (int row = 0; row < 3; ++row) {
      if (!std::getline(in, line))
        throw std::runtime_error(std::string("the file ends inside the ") +
                                 block + " block, at line " +
                                 std::to_string(row + 1) + " of 3");
      const std::vector<std::string> t = tokens_of(line);
      if (t.size() != 3)
        throw std::runtime_error(
            std::string(block) + " line " + std::to_string(row + 1) + ", '" +
            tadah::core::trim(line) + "', has " + std::to_string(t.size()) +
            (t.size() == 1 ? " field" : " fields") +
            (b == 0 ? ": each cell line is one lattice vector, 3 numbers"
                    : ": each stress line is one row of the stress tensor, "
                      "3 numbers"));
      for (int k = 0; k < 3; ++k) fields[9 * b + 3 * row + k] = t[k];
    }
  }
  return fields;
}

}  // namespace detail

int Structure::read(std::ifstream &ifs) {
  std::string line;
  // Ignore extra empty lines (a line of only whitespace, '\r' included, is
  // empty)
  std::getline(ifs, line);
  if (tadah::core::is_blank_line(line)) return 1;

  // The first line read is used as a label
  label = line;

  // Every value must be a finite number: a token the stream cannot read
  // would leave it failed short of the end of the file. Errors name the
  // structure; StructureDB::add adds the file and the position.
  try {
  // The second line is the energy, 'energy T', or the weights 'eweight
  // fweight sweight' followed by a line 'energy [T]'.
  std::getline(ifs, line);
  std::vector<std::string> tok = tokens_of(line);
  if (tok.size() == 3) {
    eweight = weight_value(tok[0], "eweight");
    fweight = weight_value(tok[1], "fweight");
    sweight = weight_value(tok[2], "sweight");
    std::getline(ifs, line);
    tok = tokens_of(line);
    if (tok.empty() || tok.size() > 2)
      throw std::runtime_error("the line after the weights, '" + line +
                               "', is not 'energy' or 'energy T'");
    energy = finite_value(tok[0], "energy");
    if (tok.size() == 2) T = finite_value(tok[1], "temperature");
  }
  else if (tok.size() == 2) {
    energy = finite_value(tok[0], "energy");
    T = finite_value(tok[1], "temperature");
  }
  else if (tok.size() == 1) {
    energy = finite_value(tok[0], "energy");
  }
  else {
    throw std::runtime_error("the line after the label, '" + line +
                             "', is not 'energy', 'energy T' or 'eweight "
                             "fweight sweight'");
  }

  // The 3x3 cell matrix, then the 3x3 stress matrix: three lines of three
  // numbers each (detail::read_cell_stress_lines, which next_structure and
  // StructureDB::find_unique_elements read the same way).
  const std::array<std::string, 18> block = detail::read_cell_stress_lines(ifs);
  for (int b = 0; b < 2; ++b) {
    const std::string what = b == 0 ? "cell" : "stress";
    tadah::core::Matrix3d &m = b == 0 ? cell : stress;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        m(i, j) = finite_value(block[9 * b + 3 * i + j],
                               what + " entry (" + std::to_string(i + 1) +
                                   ", " + std::to_string(j + 1) + ")");
  }

  // Clear existing atoms
  atoms.clear();

  // Read in each atom: 'Element px py pz fx fy fz'
  while(std::getline(ifs, line)) {
    if (tadah::core::is_blank_line(line)) break;  // '\r' (CRLF) included

    tok = tokens_of(line);
    if (tok.size() < 7)
      throw std::runtime_error("the atom line '" + line +
                               "' is not 'Element px py pz fx fy fz'");
    double v[6];
    static const char *const names[6] = {"px", "py", "pz", "fx", "fy", "fz"};
    for (int k = 0; k < 6; ++k)
      v[k] = finite_value(tok[k + 1], std::string(names[k]) + " of the atom line '" + line + "'");
    tadah::core::Element e;
    try {
      e = tadah::core::PeriodicTable::find_by_symbol(tok[0]);
    } catch (const std::exception &) {
      throw std::runtime_error("the atom line '" + line + "' starts with '" +
                               tok[0] + "', which is not an element symbol");
    }
    atoms.push_back(Atom(e, v[0], v[1], v[2], v[3], v[4], v[5]));
  }
  } catch (const std::exception &e) {
    throw std::runtime_error("structure '" + label + "': " + e.what());
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
 * Every double (weights, energy, cell, stress, positions, forces) is
 * written with max_digits10 (17) significant digits, so read() returns
 * bit-identical values; columns are separated by at least one space. The
 * stream's formatting state is restored before returning.
 *
 * Example usage:
 *     std::ofstream ofs("my_structure_output.dat");
 *     st.save(ofs);
 */
void Structure::save(std::ofstream &ofs) const {
  const std::ios_base::fmtflags flags = ofs.flags();
  const std::streamsize precision = ofs.precision();
  ofs << std::defaultfloat
      << std::setprecision(std::numeric_limits<double>::max_digits10);
  auto col = [&ofs](double v) {
    ofs << ' ' << std::right << std::setw(w - 1) << v;
  };

  ofs << label << '\n';
  ofs << eweight << ' ' << fweight << ' ' << sweight << '\n';
  ofs << energy << '\n';
  for (int i = 0; i < 3; ++i) {
    col(cell(i,0)); col(cell(i,1)); col(cell(i,2));
    ofs << '\n';
  }

  for (int i = 0; i < 3; ++i) {
    col(stress(i,0)); col(stress(i,1)); col(stress(i,2));
    ofs << '\n';
  }

  for (const Atom &a : atoms) {
    ofs << std::left << std::setw(2) << a.symbol;
    col(a.position(0)); col(a.position(1)); col(a.position(2));
    col(a.force(0)); col(a.force(1)); col(a.force(2));
    ofs << '\n';
  }
  ofs.flags(flags);
  ofs.precision(precision);
}

/**
 * Saves the current structure data to a file.
 * Reason: Directly writes all attributes of Structure to a user-specified file path.
 *
 * Throws std::runtime_error if the file cannot be opened or written.
 *
 * Example usage:
 *     st.save("my_structure_output.dat");
 *
 * Warning: Overwrites the file if it already exists.
 */
void Structure::save(const std::string fn) const {
  std::ofstream ofs(fn);
  if (!ofs.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + fn);
  }
  save(ofs);
  ofs.close();
  if (!ofs) {
    throw std::runtime_error("Error writing structure to file: " + fn);
  }
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
  for (size_t x=0; x<nn->idx[j].size(); ++x)
  {
    if (nn->idx[j][x]==i && nn->shift[j][x]==shift_ijj) return x;
  }
  // NNFinder's tables are symmetric, so this is a table built or changed
  // elsewhere; returning idx[j].size() would index past atom j's list.
  throw std::logic_error(
      "Structure::get_nn_iindex: neighbour " + std::to_string(jj) +
      " of atom " + std::to_string(i) + " (atom " + std::to_string(j) +
      ") has no reverse entry in atom " + std::to_string(j) +
      "'s list; the neighbour list is not symmetric");
}

/**
 * Returns the volume of the cell, |a0 . (a1 x a2)| with a_i = cell.row(i).
 * The absolute value: a left-handed basis (CASTEP accepts one, and a
 * POSCAR can hold one) describes the same lattice and has the same volume.
 */
double Structure::get_volume() const {
  return std::fabs(cell.row(0)*(cell.row(1).cross(cell.row(2))));
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
 * Returns the virial pressure -trace(stress)/3 in the stress unit
 * (eV/A^3): positive under compression, since the stored stress is the
 * Cauchy stress, positive under tension.
 */
double Structure::get_virial_pressure() const {
  return -stress.trace() / 3.0;
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
  return vpress + natoms() * kB * T / get_volume();  // + N k_B T / V
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

namespace {
/* Refuses a cell whose lattice vectors are zero, non-finite or (nearly)
 * coplanar: positions are compared through its inverse. The criterion is
 * the one of the POSCAR exporter: |det| <= 1e-12 |a0| |a1| |a2|. */
void require_regular_cell(const Structure &st) {
  const tadah::core::Vec3d a0 = st.cell.row(0), a1 = st.cell.row(1),
                           a2 = st.cell.row(2);
  const double det = a0 * a1.cross(a2);
  if (!std::isfinite(det) ||
      !(std::fabs(det) > 1e-12 * a0.norm() * a1.norm() * a2.norm()))
    throw std::invalid_argument(
        "Structure::is_the_same: structure '" + st.label + "' (" +
        std::to_string(st.natoms()) +
        " atoms) has a singular cell (zero, non-finite or coplanar lattice "
        "vectors); its positions cannot be compared through the cell");
}

/* The cell as is_the_same compares positions through it: its inverse, the
 * spacings h_k of its lattice planes (1 / |column k of the inverse|) and
 * the length of its shortest lattice vector. */
struct ComparisonCell {
  tadah::core::Matrix3d cell, inv;
  double h[3];
  double shortest;
};

ComparisonCell comparison_cell(const Structure &st) {
  ComparisonCell c;
  c.cell = st.cell;
  c.inv = st.cell.inverse();
  for (int k = 0; k < 3; ++k)
    c.h[k] = 1.0 / std::sqrt(c.inv(0, k) * c.inv(0, k) + c.inv(1, k) * c.inv(1, k) +
                             c.inv(2, k) * c.inv(2, k));
  // A lattice vector n.cell no longer than L has |n_k| <= L / h_k, with L
  // the shortest row: enumerate those n.
  double L = std::numeric_limits<double>::infinity();
  for (int r = 0; r < 3; ++r) L = std::min(L, st.cell.row(r).norm());
  long N[3];
  double count = 1.0;
  for (int k = 0; k < 3; ++k) {
    N[k] = static_cast<long>(std::floor(L / c.h[k] * (1.0 + 1e-12)));
    count *= 2.0 * N[k] + 1.0;
  }
  if (!(count <= 1e6))
    throw std::invalid_argument(
        "Structure::is_the_same: the cell of structure '" + st.label +
        "' is too skewed to compare positions through it (write it with a "
        "reduced basis)");
  c.shortest = L;
  for (long a = -N[0]; a <= N[0]; ++a)
    for (long b = -N[1]; b <= N[1]; ++b)
      for (long d = -N[2]; d <= N[2]; ++d) {
        if (a == 0 && b == 0 && d == 0) continue;
        double v2 = 0.0;
        for (int x = 0; x < 3; ++x) {
          const double v = a * st.cell(0, x) + b * st.cell(1, x) + d * st.cell(2, x);
          v2 += v * v;
        }
        c.shortest = std::min(c.shortest, std::sqrt(v2));
      }
  return c;
}

/* Whether d minus some lattice vector of c lies within thr in every
 * Cartesian component. Every lattice vector n.cell for which that holds has
 * |n_k - f_k| <= sqrt(3) thr / h_k (f the fractional coordinates of d), so
 * those n are all tried. */
bool within_through(const ComparisonCell &c, const double d[3], double thr) {
  double f[3];
  long lo[3], hi[3];
  for (int k = 0; k < 3; ++k) {
    f[k] = d[0] * c.inv(0, k) + d[1] * c.inv(1, k) + d[2] * c.inv(2, k);
    const double r = std::sqrt(3.0) * thr / c.h[k] + 1e-9 * (1.0 + std::fabs(f[k]));
    lo[k] = static_cast<long>(std::ceil(f[k] - r));
    hi[k] = static_cast<long>(std::floor(f[k] + r));
  }
  for (long a = lo[0]; a <= hi[0]; ++a)
    for (long b = lo[1]; b <= hi[1]; ++b)
      for (long e = lo[2]; e <= hi[2]; ++e) {
        bool within = true;
        for (int x = 0; x < 3 && within; ++x)
          within = std::fabs(d[x] - (a * c.cell(0, x) + b * c.cell(1, x) +
                                     e * c.cell(2, x))) <= thr;
        if (within) return true;
      }
  return false;
}

/* Refuses a threshold at which two images of one position could both lie
 * within it: two such images differ by a lattice vector no longer than
 * 2 sqrt(3) thr. */
void require_unambiguous_threshold(const Structure &st, const ComparisonCell &c,
                                   double thr) {
  const double limit = c.shortest / (2.0 * std::sqrt(3.0));
  if (!(thr < limit)) {
    std::ostringstream oss;
    oss << "Structure::is_the_same: the threshold " << thr
        << " A is too large for structure '" << st.label
        << "': it must be below lambda / (2 sqrt 3) = " << limit
        << " A, lambda = " << c.shortest
        << " A its shortest lattice vector, so that at most one image of a "
           "position lies within it";
    throw std::invalid_argument(oss.str());
  }
}
}  // namespace

/**
 * Checks if two structures match in cell vectors, number of atoms, species, and coordinates,
 * ignoring order of atoms and not comparing energies, forces, or labels.
 * Reason: More lenient check to see if structures represent the "same" geometry.
 *
 * Each atom must match a distinct atom of `st` of the same element whose
 * position differs by a lattice vector, of either structure's cell, plus at
 * most `thr` in every Cartesian component; every such lattice vector is
 * tried (within_through). The structures are the same when such a
 * one-to-one matching of all atoms exists; it is searched for with
 * augmenting paths (Kuhn's algorithm), so the result depends neither on the
 * order of the atoms nor on which structure is called. Throws
 * std::invalid_argument when either cell is singular, or `thr` is negative,
 * not finite or so large that two images of a position could both match.
 */
bool Structure::is_the_same(const Structure& st, double thr) const {
  require_regular_cell(*this);
  require_regular_cell(st);
  if (!(thr >= 0.0) || !std::isfinite(thr))
    throw std::invalid_argument("Structure::is_the_same: the threshold must be a "
                                "finite number of at least 0");
  const ComparisonCell ca = comparison_cell(*this), cb = comparison_cell(st);
  require_unambiguous_threshold(*this, ca, thr);
  require_unambiguous_threshold(st, cb, thr);
  bool result =
    cell.isApprox(st.cell, thr) &&
    (natoms() == st.natoms());

  if (!result) return false;

  const size_t n = natoms();
  const size_t none = static_cast<size_t>(-1);

  // candidates[i]: the atoms of st that atom i may match.
  std::vector<std::vector<size_t>> candidates(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (!(static_cast<const tadah::core::Element &>(atoms[i]) ==
            static_cast<const tadah::core::Element &>(st.atoms[j])))
        continue;
      double d[3];
      for (size_t x = 0; x < 3; ++x)
        d[x] = st.atoms[j].position[x] - atoms[i].position[x];
      if (within_through(ca, d, thr) || within_through(cb, d, thr))
        candidates[i].push_back(j);
    }
    if (candidates[i].empty()) return false;
  }

  // Kuhn's algorithm, with a breadth-first search for each augmenting path
  // (no recursion). If atom i has no augmenting path, no matching covers
  // every atom.
  std::vector<size_t> match_of_this(n, none), match_of_st(n, none);
  std::vector<size_t> reached_from(n), queue;
  std::vector<char> seen(n);
  queue.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    std::fill(seen.begin(), seen.end(), 0);
    queue.assign(1, i);
    size_t free_j = none;
    for (size_t q = 0; q < queue.size() && free_j == none; ++q) {
      const size_t u = queue[q];
      for (size_t j : candidates[u]) {
        if (seen[j]) continue;
        seen[j] = 1;
        reached_from[j] = u;
        if (match_of_st[j] == none) {
          free_j = j;
          break;
        }
        queue.push_back(match_of_st[j]);
      }
    }
    if (free_j == none) return false;
    // Flip the path back to i: each atom on it takes the atom of st it
    // reached, and its previous match goes to the atom before it.
    for (size_t j = free_j; j != none;) {
      const size_t u = reached_from[j];
      const size_t previous = match_of_this[u];
      match_of_this[u] = j;
      match_of_st[j] = u;
      j = previous;
    }
  }
  return true;
}

/**
 * Moves the file stream to skip over the next structure.
 * Reason: Used to step through multiple structures in a single file.
 *
 * Returns false at the end of the file; otherwise true, with natoms the
 * number of atoms of the skipped structure (0 for one without atoms, which
 * is a structure as for read(), not the end of the file).
 */
bool Structure::next_structure(std::ifstream &ifs, size_t &natoms) {
  std::string line;
  // Skip the structure label line, after any blank lines (a line of only
  // whitespace, '\r' included, is blank, as in Structure::read); false at
  // the end of the file.
  do {
    if (!std::getline(ifs, line)) return false;
  } while (tadah::core::is_blank_line(line));
  const std::string label = line;

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

  // The cell and stress lines, three of three fields each, as read() reads
  // them.
  try {
    (void)detail::read_cell_stress_lines(ifs);
  } catch (const std::exception &e) {
    throw std::runtime_error("structure '" + label + "': " + e.what());
  }

  natoms = 0;
  while(std::getline(ifs, line)) {
    if (tadah::core::is_blank_line(line)) break;
    natoms++;
  }
  return true;
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
 * Every number is written with `prec` SIGNIFICANT digits (%.{N}g, NUMERIC
 * of the data engines): the default 18 reads back bit-identically. Each
 * number column is prec + 7 characters wide (the widest rendering,
 * "-d.<N-1 digits>e-308"), so columns stay aligned.
 *
 * Example usage:
 *     st.dump_to_file(std::cout);
 */
void Structure::dump_to_file(std::ostream &file, size_t prec) const {
  const int digits = static_cast<int>(
      std::min<size_t>(std::max<size_t>(prec, 1), tadah::core::kMaxSigDigits));
  const int width = digits + 7;
  char buf[tadah::core::kReformatBufferSize];
  // One number, optionally right-aligned in `w` characters (no allocation).
  auto put = [&](double v, int w) {
    const std::size_t len =
        tadah::core::format_sig_chars(v, digits, buf, sizeof buf);
    if (w > 0) file << std::setw(w);
    file << std::string_view(buf, len);
  };
  auto row = [&](const tadah::core::Matrix3d &m, int i) {
    put(m(i,0), width); file << " ";
    put(m(i,1), width); file << " ";
    put(m(i,2), width); file << " " << '\n';
  };

  file << label << '\n';
  put(eweight, 0); file << " ";
  put(fweight, 0); file << " ";
  put(sweight, 0); file << '\n';
  put(energy, 0); file << " ";
  put(T, 0); file << '\n';

  for (int i = 0; i < 3; ++i) row(cell, i);
  for (int i = 0; i < 3; ++i) row(stress, i);

  for (const auto &a : atoms) {
    file << std::setw(2) << a.symbol << " ";
    put(a.position[0], width); file << " ";
    put(a.position[1], width); file << " ";
    put(a.position[2], width); file << " ";
    put(a.force[0], width); file << " ";
    put(a.force[1], width); file << " ";
    put(a.force[2], width); file << '\n';
  }
  file << '\n';
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
  if (!file.is_open())
    throw std::runtime_error("Could not open file for writing: " + filepath);
  dump_to_file(file, prec);
  file.close();
  if (file.fail())
    throw std::runtime_error("Could not write the file: " + filepath);
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
