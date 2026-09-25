#ifndef STRUCTURE_DB_h
#define STRUCTURE_DB_h

#include "tadah/core/utils.h"
#include <tadah/core/element.h>
#include <tadah/core/context.h>
#include <tadah/mlip/structure.h>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>

namespace tadah {
namespace mlip {
/**
 * Container for a collection of Structure(s).
 *
 * Usage example:
 *
 *     # Default constructor creates an empty StructureDB object.
 *     StructureDB stdb;
 *     # Add structures from the file "db.dat"
 *     stdb.add("db.dat");
 *
 * Usage example:
 *
 *     # Load all structure files listed in a Context file
 *     Context context("context");
 *     StructureDB stdb(context);
 *
 * Usage example:
 *
 *     # Print this object summary using streams:
 *     std::cout << stdb;
 *
 *     # Print 3rd Atom data from the 1st Structure
 *     std::cout << stdb(0,2);
 *
 * @see Structure NNFinder
 *
 */
struct StructureDB {
  std::vector<Structure> structures;

  /** Create an empty StructureDB object. */
  StructureDB();

  /** Create this object and load structures
     *  listed in the context file
     *
     * \note
     *      Required Context key: \ref DBFILE
     */
  StructureDB(const tadah::core::Context &context);

  /** Add structures listed in the context file
     *
     *	\note
     *      Required Context key: \ref DBFILE
     *
     */
  void add(const tadah::core::Context &context);

  /** Add all structures from a file */
  void add(const std::string fn);

  /** Add N structures from a file beginning from first index.
     *
     * Indexing starts from zero, i.e.
     * first=0 corresponds to the first structure in the file.
     *
     * In the case of N greater than the number of available 
     * structures, this method will load only the available ones.
     *
     * Return the number of loaded structures.
     */
  int add(const std::string fn, size_t first, int N=1);

  /** Add a single Structure object to this container */
  void add(const Structure &s);

  /** Add all structure from other Structure object to this container */
  void add(const StructureDB &stdb);

  /** remove i-th Structure object from this container */
  void remove(size_t i);

  /** \return number of structures held by this object */
  size_t size() const;

  /** \return number of structures in the n-th DBFILE
     *
     * n={0,...,number of DBFILEs-1}
     */
  size_t size(size_t n) const;

  /** \return reference to the s-th structure
     *
     * Usage example:
     *
     *     # Get reference to the 2nd structure held be this object
     *     # and bind it to Structure st.
     *     Structure &st = st(1);
     */
  Structure& operator()(size_t s);
  const Structure& operator()(size_t s) const;

  /** \return reference to the a-th atom in the s-th structure
     *
     * Usage example:
     *
     *     # Get reference to the 5th atom in the 3rd structure
     *     # held by this object and bind it to the atom object.
     *     Atom &atom = st(2,4);
     */
  Atom& operator()(size_t s, size_t a);

  /** Print this object summary to the stream */
  friend std::ostream& operator<<(std::ostream& os, const StructureDB& stdb)
  {
    os << stdb.summary();
    return os;
  }
  std::string summary() const;

  /** Store  indices for each dataset.
     *
     * e.g. if 3 datasets of sizes 11,13,15
     * dbidx=={0,11,24,39}
     */
  std::vector<size_t> dbidx;

  /** Calculate total number of atoms stored by this object. */
  size_t calc_natoms() const;

  /** Calculate total number of atoms in the n-th DBFILE.
     *
     * n={0,...,number of DBFILEs-1}
     */
  size_t calc_natoms(size_t n) const;

  /** Return unique elements for all Structures. */
  std::set<tadah::core::Element> get_unique_elements() const;

  /** Find unique elements in provided Context file */
  static std::set<tadah::core::Element> find_unique_elements(const tadah::core::Context &c);

  /** Find unique elements in provided file */
  static std::set<tadah::core::Element> find_unique_elements(const std::string &fn);

  /** Count number of structures and atoms in all datasets from the Context file. */
  static std::pair<int,int> count(const tadah::core::Context &c);

  /** Count number of structures and atoms in a single dataset. */
  static std::pair<int,int> count(const std::string fn);

  void clear_nn();

  /** Check consistency of the ATOM key. */
  static void check_atom_key(tadah::core::Context &context, std::set<tadah::core::Element> &unique_elements);

  /** Check consistency of the WATOM key. Add if missing*/
  static void check_watom_key(tadah::core::Context &context, std::set<tadah::core::Element> &unique_elements);

  // Methods to enable range-based for loop
  std::vector<Structure>::iterator begin();
  std::vector<Structure>::iterator end();
  std::vector<Structure>::const_iterator begin() const;
  std::vector<Structure>::const_iterator end() const;

  /** Method to dump class content to a file (dataset format).
   *
   * Every number is written with `prec` SIGNIFICANT digits (%.{N}g); the
   * default 18 reads back bit-identically.
   */
  void dump_to_file(const std::string& filepath, bool append, size_t prec=18) const;

  /** Significant digits for the dataset files the `tadah data` engines
   *  write: NUMERIC from @p ctx, looked up without default insertion
   *  (absent -> 18, exact). An explicit NUMERIC below 17 is honoured, with
   *  a WARNING (the written dataset then differs from the one in memory).
   */
  static size_t dataset_digits(const tadah::core::Context &ctx);

  /** Significant digits for the structure exports of `tadah data write`
   *  that write %.{N}g reals (LAMMPS data file, VASP POSCAR): the same
   *  resolution as dataset_digits(), and the same WARNING below 17, naming
   *  @p file_noun (e.g. "LAMMPS data file"). Not for the CASTEP cell
   *  export, whose NUMERIC is decimal places.
   */
  static size_t export_digits(const tadah::core::Context &ctx,
                              const std::string &file_noun);

  // Public method that reads the file, counts blocks and line counts per block,
  // then prints the results to std::cout.
  void parseFile(const std::string& filename);

  void clear();

private:
  // Checks if a line is empty or contains only whitespace
  bool isBlankLine(const std::string& line) const;

};
} // namespace mlip
} // namespace tadah
#endif
