#ifndef STRUCTURE_h
#define STRUCTURE_h

#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_table.h>
#include <tadah/mlip/stream_format_guard.h>
#include <tadah/core/core_types.h>
#include <tadah/core/periodic_table.h>

#include <array>
#include <istream>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <set>
#include <unordered_map>

namespace tadah {
namespace mlip {

/**
 * Container for a collection of Atom(s).
 *
 * Holds list of Atom(s), cell dimensions, system energy and stress tensor.
 *
 * Pass this object to NNFinder to build a list of nearest
 * neighbour positions for every Atom in the list.
 *
 * Usage example:
 *
 *     # Default constructor creates an empty Structure object.
 *     # All attributes are left uninitialised
 *     Structure st;
 *
 * Usage example:
 *
 *     # Print object summary with streams:
 *     std::cout << st;
 *
 *     # Print 2nd atom data
 *     std::cout << st(2)
 *
 *     # Print stress tensor
 *     std::cout << st.stress
 *
 * @see Atom StructureDB NNFinder
 */
struct Structure {

  /** Create an empty Structure object.
   *
   * All class attributes are left uninitialised.
   */
  Structure();

  /** Text label for this structure. */
  std::string label;

  /** Total energy of this structure */
  double energy;

  /** Lattice vectors, in Angstrom: row i is lattice vector a_i, so an
   *  atom at fractional coordinates f is at r = f0 a0 + f1 a1 + f2 a2
   *  (cell.transpose() * f). The basis may be left- or right-handed;
   *  get_volume() is |a0 . (a1 x a2)|. The dataset format writes the rows
   *  as the three cell lines. The cell may be thinner than any cutoff and
   *  the atoms need not lie inside it (fractional coordinates outside
   *  [0, 1) are fine): NNFinder finds every periodic image. */
  tadah::core::Matrix3d cell;

  /** Stress tensor: the Cauchy stress in eV/Angstrom^3, positive under
   *  tension (sigma = (1/V) dE/d(strain), what the model predicts; the
   *  negative of VASP's 'in kB' tensor, and of the pressure LAMMPS
   *  reports). The rows are those of the 3x3 tensor; it is symmetric. */
  tadah::core::stress_type stress;

  /** Container for atoms which belong to this structure */
  std::vector<Atom> atoms;

  /** Temperature of this structure.
   *
   * Default is 0.0
   */
  double T=0;

  /** Neighbour list: every periodic image of every atom within its
   *  cutoff (NNTable::cutoff), as NNFinder builds it; nullptr until
   *  NNFinder fills it. Atoms may lie outside the cell: shifts are
   *  relative to the stored positions. */
  mlip::NNTablePtr nn;

  /** Weighting parameter for energy used during training.
   *
   * Default is 1.0
   */
  double eweight=1.0;

  /** Weighting parameter for forces used during training.
   *
   * Default is 1.0
   */
  double fweight=1.0;

  /** Weighting parameter for stress used during training.
   *
   * Default is 1.0
   */
  double sweight=1.0;


  /** @return a reference to the i-th Atom.
   *
   * Usage example:
   *
   *     # Get a reference to the 3rd Atom in the st Structure
   *     const Atom &atom = st(2);
   */
  const Atom& operator()(const size_t i) const;

  /** Add Atom a to this structure. */
  void add_atom(Atom a);

  /** Remove an Atom in the i-th position from this structure. */
  void remove_atom(const size_t i);

  /** @return  a number of atoms in this structure. */
  size_t natoms() const;

  /** @return A map containing natoms per type Z. */
  std::unordered_map<int,size_t> get_natoms_map() const;

  /** @return volume of this structure. */
  double get_volume() const;

  /** @return density of this structure in g/cm^3 */
  double get_density() const;

  /** @return temperature of this structure in K */
  double get_temperature() const;

  /** @return the virial pressure, -trace(stress)/3: positive under
   *  compression, since the stored stress is the Cauchy stress (positive
   *  under tension).
   *
   *  Units: those of the stress, eV/Angstrom^3.
   */
  double get_virial_pressure() const;

  /** @return pressure for a given temperature.
   *
   *  The calculated pressure contains both ideal gas
   *  and virial pressures.
   *
   *  Default units Ev/Angstrom^3.
   *
   *  @param kB Boltzmann Constant. Default in ev/K.
   *
   *  For T=0 returns virial pressure only.
   */
  double get_pressure(const double T, const double kB=8.617333262145e-5) const;

/**
 * Return the position of the n-th neighbour of atom i: the position of
 * atom nn->idx[i][n] plus its periodic image displacement nn->disp[i][n]
 * = s0 a0 + s1 a1 + s2 a2 for the integer shift s = nn->shift[i][n]
 * (a_k = cell.row(k); NNFinder::populate_disp fills disp).
 *
 * Returns by value (tadah::core::Vec3d) so we don't reference a temporary.
 */
  tadah::core::Vec3d nn_pos(const size_t i, const size_t n) const;

  /** @return a number of nearest neighbours of the i-th Atom. */
  size_t nn_size(const size_t i) const;

  /** Read structure from the stream.
   *
   * If a stream contains more than one structure,
   * the first structure will be read.
   *
   * A line of only whitespace (a CRLF file's '\r' included) is empty: it
   * ends the structure. Fields after the seventh of an atom line are
   * ignored.
   *
   * @returns 0 if success, 1 otherwise
   * @throws std::runtime_error when a value is not a finite number or an
   *         atom line has fewer than the seven fields 'Element px py pz fx
   *         fy fz'; the message names the structure's label and the bad
   *         text.
   */
  int read(std::ifstream &ifs);

  /** Read a structure from the file.
   *
   * If a file contains more than one structure,
   * the first structure will be read.
   */
  void read(const std::string fn);

  /** Save structure to the stream. */
  void save(std::ofstream &ofs) const;

  /** Save structure to the file.
   *
   * @warning
   *      This will overwrite any existing data in the fn file.
   */
  void save(const std::string fn) const;

  /**  Return corresponding ii index of i-j (jj) atom interaction.
   *
   *  The ii index is used with nearest neighbour lists.
   *  i,j are atom global indices; ii,jj are neighbour lists indices.
   *
   *  This method is required to fully specify interacton of two atoms
   *  when predicting force on the i-th atom. In particular in the case when
   *  atom i-j interaction is specified more than once. e.g. one i-j interaction
   *  in the main box and another one coming from the periodic image.
   *  The critical condition for this to happen for the square box is:
   *  rcut > box length / 2.
   *
   *  e.g.
   *  To get a force on atom i. We loop over nn of i to obtain jj indices.
   *  For many-body potentials we also need to know the density for atom at jj,
   *  hence we need j index. This is stored by:
   *
   *      j=nn->idx[a][jj]
   *
   *  Moreover we need to know derivatives (descriptor derivatives wrt to rij)
   *  at both i and j atoms. However if jj atom is outside of the main box
   *  we do not calculate derivative for this atom but we can use derivative
   *  for the j-ii interaction. Hence we need to find ii index here...
   *
   *
   *      ii = get_nn_iindex(i,j,jj)
   *
   *  phew...
   *
   *  Throws std::logic_error when atom j's list has no entry (i, −shift):
   *  never for a table NNFinder built, which is symmetric.
   */
  size_t get_nn_iindex(const size_t i, const size_t j, const size_t jj) const;

  /** Print structure summary using streams. Leaves the format of @p os
   *  (flags, precision, width, fill) as it found it. */
  friend std::ostream& operator<<(std::ostream& os, const Structure& st)
  {
    const StreamFormatGuard keep(os);
    os << "Structure  :    " << std::left << st.label << std::endl;
    os << "Weights    :    " << std::left
      << "E: "    << st.eweight
      << " | F: " << std::left << st.fweight
      << " | S: " << std::left << st.sweight << std::endl;
    os << "Atoms      :    " << std::left << st.atoms.size() << " | ";
    for (const auto &el: st.get_unique_elements()) {  os << " " << el.symbol << std::left; }
    os << std::endl;
    os << "Energy     :" << std::right << std::setw(w)
      << std::setprecision(p) << std::fixed << st.energy << "     "
       << "Temp   :" << std::right << std::setw(w)
      << std::setprecision(p) << std::fixed << st.get_temperature() << std::endl;
    os << "Volume     :" << std::right << std::setw(w)
      << std::setprecision(p) << std::fixed << st.get_volume() << "     "
       << "Density:" << std::right << std::setw(w)
      << std::setprecision(p) << std::fixed << st.get_density() << std::endl;
    os << "Cell       :" << std::endl;
    for (int i=0; i<3; ++i) {
      os << std::right << std::fixed << std::setw(w+12)
        << std::setprecision(p) << st.cell(i,0);
      os << std::right << std::fixed << std::setw(w)
        << std::setprecision(p) << st.cell(i,1);
      os << std::right << std::fixed << std::setw(w)
        << std::setprecision(p) << st.cell(i,2);
      os << std::endl;
    }
    os << "Stress     :" << std::endl;
    for (int i=0; i<3; ++i) {
      os << std::right << std::fixed << std::setw(w+12)
        << std::setprecision(p) << st.stress(i,0);
      os << std::right << std::fixed << std::setw(w)
        << std::setprecision(p) << st.stress(i,1);
      os << std::right << std::fixed << std::setw(w)
        << std::setprecision(p) << st.stress(i,2);
      os << std::endl;
    }
    return os;
  }
  /** Return true if both structures are the same.
   *
   * This operator compares cell, stress tensor, number of atoms,
   * eweight, fweight and sweight, all atom positions and forces.
   *
   * Assumes atoms have the same order.
   *
   * It does NOT compare labels.
   */
  bool operator==(const Structure& st) const;

  /** Return true if both structures are the same.
   *
   * This method compares cell vectors, number of atoms,
   * species and atomic coordinates
   *
   * Order of atoms does not matter: the structures are the same when each
   * atom can be paired with a distinct atom of `st` of the same element
   * whose position differs by a lattice vector of either cell plus at most
   * `thr` in each Cartesian component (one-to-one, found whenever it
   * exists). The relation is symmetric.
   *
   * It does NOT compare energies, stresses, forces and labels.
   *
   * @throws std::invalid_argument when the cell of either structure is
   *         singular (zero, non-finite or coplanar lattice vectors), or
   *         `thr` is negative, not finite, or not below lambda/(2 sqrt 3),
   *         lambda the shortest lattice vector of either cell (above it two
   *         images of one position could both match), or a cell is written
   *         with so skewed a basis that finding lambda would take more than
   *         10^6 trial translations (write it with a reduced basis); the
   *         message names the structure.
   */
  bool is_the_same(const Structure& st, double thr=1e-6) const;

  /** Skip the next structure of a dataset stream without parsing it.
   *
   * Blank lines (whitespace only, '\r' included) before a label are
   * skipped, and a blank line ends the atoms, as in read(); the cell and
   * stress blocks are three lines of three fields each, as read() reads
   * them (detail::read_cell_stress_lines).
   *
   * @return false at the end of the stream, where no label is left;
   *         otherwise true, with @p natoms set to the number of atom lines
   *         of the skipped structure (0 for a structure without atoms,
   *         whose blank line comes straight after the stress lines).
   * @throws std::runtime_error naming the structure's label when its cell
   *         or stress block has another layout, or the stream ends in it.
   */
  static bool next_structure(std::ifstream &ifs, size_t &natoms);

  /* clearing == drop the shared pointer */
  void clear_nn();

  // Methods to enable range-based for loop
  std::vector<Atom>::iterator begin();
  std::vector<Atom>::iterator end();
  std::vector<Atom>::const_iterator begin() const;
  std::vector<Atom>::const_iterator end() const;

  /** Method to dump class content to a file (dataset format).
   *
   * Every number is written with `prec` SIGNIFICANT digits (%.{N}g); the
   * default 18 reads back bit-identically.
   */
  void dump_to_file(std::ostream &file, size_t prec=18) const;

  /** Method to dump class content to a file (appends; see above).
   * @throws std::runtime_error naming the file when it cannot be opened or
   *         written. */
  void dump_to_file(const std::string& filepath, size_t prec=18) const;

  /** List of chemical elements in this structure. */
  std::set<tadah::core::Element> get_unique_elements() const;

  /* Return all 3N Cartesian coordinates as a container         *
   * Container must be default-constructible, hold scalars and  *
   * provide reserve() and operator[].                          */

// --- inside struct Structure ------------------------------------
template<class Container = std::vector<double>>
Container get_coords() const
{
    Container x;
    x.reserve(3 * natoms());
    for(const auto& a : atoms)
        for(int c = 0; c < 3; ++c)
            x.push_back(static_cast<typename Container::value_type>
                        (a.position(c)));
    return x;
}

template<class Container>
void set_coords(const Container& x)
{
    if(x.size() != 3 * natoms())
        throw std::runtime_error("Structure::set_coords – size mismatch");
    std::size_t k = 0;
    for(auto& a : atoms)
        for(int c = 0; c < 3; ++c)
            a.position(c) = static_cast<double>(x[k++]);
}

template<class Scalar>
void set_coords(const std::vector<Scalar>& x)
{
    if(x.size() != 3 * natoms())
        throw std::runtime_error("size mismatch in set_coords");
    std::size_t k = 0;
    for(auto& a : atoms)
        for(int c = 0; c < 3; ++c)
            a.position(c) = x[k++];        // NO cast to double
}


  private:
    const static size_t w=13;  // controls output width
    const static size_t p=6;   // controls output precision
  

};

namespace detail {

/** The cell and stress blocks of a dataset structure, read from @p in:
 *  three cell lines (one lattice vector each), then three stress lines (one
 *  row of the stress tensor each), exactly three fields on each line (the
 *  Dataset Format). Returns the 18 fields in that order, unconverted.
 *  Structure::read, Structure::next_structure and
 *  StructureDB::find_unique_elements read the blocks with it, so every
 *  reader of a dataset agrees on where a structure's atoms start.
 *  @throws std::runtime_error naming the block and the line ('cell line 1,
 *          '...', has 9 fields: ...') or where the stream ends inside the
 *          blocks. */
std::array<std::string, 18> read_cell_stress_lines(std::istream &in);

}  // namespace detail
} // namespace mlip
} // namespace tadah
#endif
