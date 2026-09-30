#pragma once
#include <memory>
#include <vector>
#include <tadah/core/core_types.h>   // Vec3d

namespace tadah::mlip
{
/*  Immutable container for one structure’s neighbour list.
 *  After construction the vectors are never modified, therefore
 *  a single instance can be shared safely between any number of
 *  threads (via std::shared_ptr<const NNTable>).
 *
 *  NNFinder fills it; its contract (every periodic image closer than the
 *  cutoff, symmetric, sorted by (j, shift)) is documented on NNFinder. */
class NNTable
{
public:
  /** List of global indices for nearest neighbours for every atom */
    std::vector<std::vector<size_t>>            idx;   // [atom][k] → neighbour index
                                                       //
  /** Periodic image of each neighbour: an integer triple (n0, n1, n2),
   *  stored as doubles, relative to the stored atom positions.
   *
   *  Neighbour k of atom i is atom j = idx[i][k] displaced by
   *  n0 a0 + n1 a1 + n2 a2 (a_k = cell.row(k)). The positions are used as
   *  stored, wrapped into the cell or not, so an atom outside the cell
   *  gets the shift that brings its image to the right place.
   *
   *  Example in two dimensions for a wrapped atom:
   *
   *      -1, 1 | 0, 1 | 1, 1
   *      -------------------
   *      -1, 0 | 0, 0 | 1, 0
   *      -------------------
   *      -1,-1 | 0,-1 | 1,-1
   *
   *  So [0,0] corresponds to the atom in the main box and [1,0] to periodic
   *  shift in the x-direction. Obviously we use three dimensions so this
   *  corresponds to [0,0,0] and [1,0,0].
   *
   *  A cell thinner than the cutoff has several images in a direction,
   *  e.g. [2,0,0] and [-3,0,0].
   */
    std::vector<std::vector<tadah::core::Vec3d>>shift; // [atom][k] → integer (n1,n2,n3)

    /** Real-space neighbour displacement, precomputed at NN-build time as
     *  shift[i][k][0]*cell.row(0) + shift[i][k][1]*cell.row(1) + shift[i][k][2]*cell.row(2).
     *
     *  Cached so Structure::nn_pos() can return
     *      atoms[idx[i][k]].position + disp[i][k]
     *  without re-multiplying the cell on every call. The shift cost
     *  dominated the descriptor calc inner loops (~12% of CPU in perf
     *  reports). Keeping the integer-triple `shift` alongside lets
     *  downstream tests that depend on lattice-image identities keep
     *  working. */
    std::vector<std::vector<tadah::core::Vec3d>>disp;  // [atom][k] → real shift displacement

    /** Cutoff the list was built for (Å): it holds every pair closer than
     *  this. 0 for a table NNFinder did not build. A list that nn_cache
     *  keeps after the atoms or the cell moved is complete only for
     *  cutoff − skin (nn_cache::is_fresh). */
    double cutoff = 0.0;

    /** Width of the band above `cutoff` in which the list may also hold
     *  pairs (Å, about 1e-13 relative): the rounding bound of a pair
     *  distance. Consumers filter by their own cutoff. */
    double guard = 0.0;

    NNTable()  = default;
    NNTable(const NNTable&)            = default;
    NNTable& operator=(const NNTable&) = default;

    /* convenience helpers */
    size_t nn_size(size_t i) const              { return idx[i].size(); }
    const std::vector<size_t>& operator[](size_t i) const { return idx[i]; }
};
using NNTablePtr = std::shared_ptr<const NNTable>;
} // namespace tadah::mlip
