#ifndef STRUCTURE_DB_h
#define STRUCTURE_DB_h

#include "structure.h"
#include "../CORE/config/config.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>

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
 *     # Load all structure files listed in a Config file
 *     Config config("config");
 *     StructureDB stdb(config);
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
     *  listed in the config file
     *
     * \note
     *      Required Config key: \ref DBFILE
     */
    StructureDB(const Config &config);

    /** Add structures listed in the config file
     *
     *	\note
     *      Required Config key: \ref DBFILE
     *
     */
    void add(const Config &config);

    /** Add all structures from a file */
    void add(const std::string fn);

    /** Add a single Structure object to this container */
    void add(const Structure &s);

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
        os << "# of structures  :    " << std::left << stdb.structures.size() << std::endl;
        return os;
    }

    /** Store  indices for each dataset.
     *
     * e.g. if 3 datasets of sizes 11,13,15
     * dbidx=={0,11,24,39}
     */
    std::vector<size_t> dbidx;

    /** Calculate total number of atoms stored by this object */
    size_t calc_natoms() const;

    /** Calculate total number of atoms in the n-th DBFILE.
     *
     * n={0,...,number of DBFILEs-1}
     */
    size_t calc_natoms(size_t n) const;

    /** Return unique elements for all Structures */
    std::unordered_set<Element> get_unique_elements();
};
#endif
