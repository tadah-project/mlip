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
 * Nearest Neighbor Finder that constructs a full nearest neighbor list
 * for every atom in a structure, using:
 *  - a **binned** (linked-cell) approach if every cell dimension ≥ cutoff,
 *  - a **naive** approach otherwise (fallback).
 *
 * The cutoff is from context("RCUTMAX").
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
     * Compute ±N to consider for image shifts.
     */
    void num_shifts(const Structure &st, int N[3], double cutoff) const;
    /**
     * Return false if any cell dimension < cutoff => fallback to naive.
     */
    static bool check_box(const Structure &st, double cutoff);

    /**
     * Remove duplicates from the neighbor list.
     */
    void deduplicate(const Structure &st,  std::shared_ptr<NNTable> tbl) const;
    /**
     * Naive approach to build neighbor lists.
     */
    void calc_naive(const Structure &st,  std::shared_ptr<NNTable> tbl, double cutoff) const;

    /**
     * Binning-based approach. If any dimension < cutoff, fallback to naive.
     */
    void calc_binned(const Structure &st,  std::shared_ptr<NNTable> tbl, double cutoff) const;

    /**
     * Populate `tbl->disp` with real-space displacements derived from
     * `tbl->shift` and the structure's cell. Called by build() once the
     * integer-triple shifts are finalised (i.e. after deduplication).
     */
    void populate_disp(const Structure &st, std::shared_ptr<NNTable> tbl) const;

    /**
     * Construct with cutoff from context("RCUTMAX").
     */
    NNFinder() = default;

    /**
     * Build nearest neighbors for all atoms in one Structure.
     * Uses binned approach if possible, else naive fallback.
     */
    NNTablePtr build(Structure &st, double cutoff) const;

    /**
     * Build nearest neighbors for all atoms in one Structure.
     * Uses context("RCUTMAX") to determine the cutoff.
     *
     * @param st Structure to build neighbors for.
     * @param context Context to get the cutoff from.
     */
    NNTablePtr build(Structure &st, const core::Context &context) const;

    /**
     * Build nearest neighbors for all atoms in one Structure.
     * Uses binned approach if possible, else naive fallback.
     *
     * @param st Structure to build neighbors for.
     * @param cutoff Cutoff distance for neighbor search.
     */
    void calc(Structure &st, double cutoff) const;

    /**
     * Build nearest neighbors for all atoms in one Structure.
     * Uses binned approach if possible, else naive fallback.
     *
     * @param st Structure to build neighbors for.
     * @param cutoff Cutoff distance for neighbor search.
     */
    void calc(Structure &st, const core::Context &context) const;

    /**
     * Build nearest neighbors for each structure in a DB.
     */
    void calc(StructureDB &stdb, double cutoff) const;


    /**
     * Build nearest neighbors for each structure in a DB.
     * Uses context("RCUTMAX") to determine the cutoff.
     *
     * @param stdb StructureDB to build neighbors for.
     * @param context Context to get the cutoff from.
     */
    void calc(StructureDB &stdb, const core::Context &context) const;


    /* Build NN table without touching the Structure object  */
    NNTablePtr build(const Structure& st, double cutoff) const;
    NNTablePtr build(const Structure& st,
        const core::Context& ctx) const;

};

} // namespace mlip
} // namespace tadah
#endif // NN_FINDER_H


