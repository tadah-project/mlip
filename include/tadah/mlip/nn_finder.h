#ifndef NN_FINDER_H
#define NN_FINDER_H

#include <tadah/core/context.h>
#include <tadah/core/lapack.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
namespace tadah {
namespace mlip {

/**
 * @class NNFinder
 *
 * Builds the full neighbour list of every atom of a periodic structure:
 * every periodic image of every atom closer than the cutoff, for any cell
 * (orthogonal or triclinic, thicker or thinner than the cutoff) and any
 * atom positions (wrapped into the cell or not).
 *
 * Contract of the table (NNTable) for a cutoff R:
 *  - **Complete:** atom i lists every (j, s) with
 *    |r_j + s0 a0 + s1 a1 + s2 a2 − r_i| < R, where a_k = cell.row(k) and
 *    s is an integer triple (j == i when s ≠ 0). It may also list pairs up
 *    to R + NNTable::guard, the rounding bound of a distance (about
 *    1e-13 relative); consumers filter by their own cutoff.
 *  - **Relative to the stored positions:** positions are never changed;
 *    an atom outside the cell gets the shift that places its image.
 *  - **Symmetric and ordered:** (i → j, s) is listed iff (j → i, −s) is;
 *    no (j, s) twice; each atom's entries sorted by (j, s0, s1, s2).
 *  - NNTable::disp[i][k] = s·A; NNTable::cutoff = R.
 *
 * Method: the lattice basis is reduced (an integer unimodular change of
 * basis, so a skewed description of a benign lattice costs nothing), the
 * atoms are sorted into a cell list over their wrapped fractional
 * coordinates sized by the cell heights, and each unordered pair is
 * tested once with the distance computed as consumers compute it.
 *
 * Refused with std::invalid_argument: a cutoff that is not finite and
 * positive, a non-finite cell entry or position, a singular (or
 * numerically singular) cell, an atom more than 2^31 cells from the
 * origin. Refused with std::runtime_error: a cell so small for the cutoff
 * that the search would span more than 10^4 image cells (see build()).
 *
 * The context-based overloads take the cutoff from context("RCUTMAX").
 */
class NNFinder {
private:
    using NNTable      = tadah::mlip::NNTable;
    using NNTablePtr   = tadah::mlip::NNTablePtr;

    /**
     * Helper to get the max cutoff value from context.
     */
    double get_cutoff(const tadah::core::Context &context) const;

public:
    /**
     * True iff every height of the (reduced) lattice is at least @p cutoff:
     * the spacing of each family of lattice planes, not the edge lengths.
     * Then each atom meets at most one image of each atom per direction
     * within the cutoff. Throws std::invalid_argument for a singular or
     * non-finite cell.
     */
    static bool check_box(const Structure &st, double cutoff);

    /**
     * Populate `tbl->disp` with real-space displacements derived from
     * `tbl->shift` and the structure's cell:
     * disp = s0 a0 + s1 a1 + s2 a2. build() calls it; nn_cache calls it
     * when it keeps a list across a change of the cell.
     */
    void populate_disp(const Structure &st, std::shared_ptr<NNTable> tbl) const;

    NNFinder() = default;

    /**
     * Build the neighbour list of one Structure at @p cutoff (see the class
     * contract). The number of image cells searched,
     * prod_k (2 N_k + 1) with N_k = floor(cutoff / h_k) + 1 over the
     * heights h_k of the reduced lattice, may not exceed 10^4: a larger
     * search means a cell smaller than about cutoff/10 in every direction
     * (a degenerate cell, e.g. from a runaway relaxation) and throws
     * std::runtime_error.
     */
    NNTablePtr build(const Structure& st, double cutoff) const;

    /** build() with the cutoff from context("RCUTMAX"). */
    NNTablePtr build(const Structure& st,
        const core::Context& ctx) const;

    /** Same as build(const Structure&, double). */
    NNTablePtr build(Structure &st, double cutoff) const;

    /** Same as build(const Structure&, const core::Context&). */
    NNTablePtr build(Structure &st, const core::Context &context) const;

    /**
     * Build the neighbour list of one Structure and store it in `st.nn`.
     *
     * @param st Structure to build neighbors for.
     * @param cutoff Cutoff distance for neighbor search.
     */
    void calc(Structure &st, double cutoff) const;

    /**
     * Build the neighbour list of one Structure at context("RCUTMAX") and
     * store it in `st.nn`.
     */
    void calc(Structure &st, const core::Context &context) const;

    /**
     * Build the neighbour list of each structure in a DB (in parallel with
     * OpenMP). If structures fail, the error of the lowest-numbered one is
     * rethrown with the same type (std::invalid_argument or
     * std::runtime_error), prefixed "NNFinder: structure <i> of <N>
     * ('<label>'): " (1-based i).
     */
    void calc(StructureDB &stdb, double cutoff) const;

    /**
     * calc(StructureDB&, double) with the cutoff from context("RCUTMAX").
     */
    void calc(StructureDB &stdb, const core::Context &context) const;
};

} // namespace mlip
} // namespace tadah
#endif // NN_FINDER_H
