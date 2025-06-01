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
    double cutoff;      ///< The cutoff distance
    double cutoff_sq;   ///< cutoff^2 for distance checks

    /**
     * Return false if any cell dimension < cutoff => fallback to naive.
     */
    bool check_box(Structure &st);

    /**
     * Naive approach to build neighbor lists.
     */
    void calc_naive(Structure &st);


    /**
     * Remove duplicates from the neighbor list.
     */
    void deduplicate(Structure &st);

    /**
     * Binning-based approach. If any dimension < cutoff, fallback to naive.
     */
    void calc_binned(Structure &st);

    /**
     * For naive approach, compute ±N to consider for image shifts.
     */
    void num_shifts(Structure &st, int N[3]);

public:
    /**
     * Construct with cutoff from context("RCUTMAX").
     */
    NNFinder(const tadah::core::Context &context);

    /**
     * Initialize NNFinder with context.
     * Sets cutoff and cutoff_sq.
     */
    void init(const tadah::core::Context &context);

    /**
     * Build nearest neighbors for all atoms in one Structure.
     * Uses binned approach if possible, else naive fallback.
     */
    void calc(Structure &st);

    /**
     * Build nearest neighbors for each structure in a DB.
     */
    void calc(StructureDB &stdb);
};

} // namespace mlip
} // namespace tadah
#endif // NN_FINDER_H


