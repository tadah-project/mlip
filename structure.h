#ifndef STRUCTURE_h
#define STRUCTURE_h

#include "atom.h"
#include "../CORE/core_types.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_set>

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

    /** Lattice vectors. */
    Matrix3d cell;

    /** Virial stress tensor. */
    stress_type stress;

    /** Container for atoms which belong to this structure */
    std::vector<Atom> atoms;

    /**
     * Container for nearest neighbour atoms for every atom in the structure.
     */
    std::vector<std::vector<Atom>> near_neigh_atoms;

    /** Periodic image flag for neigherest neighbours.
     *
     * Indicate which periodic image the neigbouring atom belongs to.
     *
     * Example in two dimensions for illustration purposes:
     *
     *      -1, 1 | 0, 1 | 1, 1
     *      -------------------
     *      -1, 0 | 0, 0 | 1, 0
     *      -------------------
     *      -1,-1 | 0,-1 | 1,-1
     *
     * So [0,0] corresponds to the atom in the main box and [1,0] to periodic
     * shift in the x-direction. Obviously we use three dimensions so this
     * corresponds to [0,0,0] and [1,0,0].
     *
     * Note that there might be more than one periodic image in each direction.
     * This depends on the box size and the cutoff distance used.
     */
    std::vector<std::vector<Vec3d>> near_neigh_shift;

    /** List of global indices for nearest neighbours for every atom */
    std::vector<std::vector<size_t>> near_neigh_idx;

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

    /** List of chemical elements in this structure. */
    std::unordered_set<Element> unique_elements;

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

    /** @return volume of this structure. */
    double get_volume() const;

    /** @return virial pressure calculated from the stress tensor.
     *
     *  Units: energy/distance^3
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

    /** @return position of the n-th nearest neighbour of the i-th Atom. */
    const Vec3d& nn_pos(const size_t i, const size_t n) const;

    /** @return a number of nearest neighbours of the i-th Atom. */
    size_t nn_size(const size_t i) const;

    /** Read structure from the stream.
     *
     * If a stream contains more than one structure,
     * the first structure will be read.
     *
     * @returns 0 if success, 1 otherwise
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
     *  The ii index is used with nearest neighour lists.
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
     *      j=near_neigh_idx[a][jj]
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
     */
    size_t get_nn_iindex(const size_t i, const size_t j, const size_t jj) const;

    /** Print structure summary using streams. */
    friend std::ostream& operator<<(std::ostream& os, const Structure& st)
    {
        os << "Structure  :    " << std::left << st.label << std::endl;
        os << "Weights    :    " << std::left
           <<  "E: "<< st.eweight
           << " | F: " << std::left << st.fweight
           << " | S: " << std::left << st.sweight << std::endl;
        os << "# atoms    :    " << std::left << st.atoms.size() << std::endl;
        os << "Energy     :" << std::right << std::setw(w)
            << std::setprecision(p) << std::fixed << st.energy << std::endl;
        os << "Cell       :" << std::endl;
        for (int i=0; i<3; ++i) {
            os << std::right << std::fixed << std::setw(w+10)
                << std::setprecision(p) << st.cell(i,0);
            os << std::right << std::fixed << std::setw(w)
                << std::setprecision(p) << st.cell(i,1);
            os << std::right << std::fixed << std::setw(w)
                << std::setprecision(p) << st.cell(i,2);
            os << std::endl;
        }
        os << "Stress     :" << std::endl;
        for (int i=0; i<3; ++i) {
            os << std::right << std::fixed << std::setw(w+10)
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
     * It does NOT compare labels/
     */
    bool operator==(const Structure& st) const;

    private:
    const static size_t w=15;  // controls output width
    const static size_t p=6;   // controls output precision

};
#endif
