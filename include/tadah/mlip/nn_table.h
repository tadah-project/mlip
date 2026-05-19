#pragma once
#include <memory>
#include <vector>
#include <tadah/core/core_types.h>   // Vec3d

namespace tadah::mlip
{
/*  Immutable container for one structure’s neighbour list.
 *  After construction the vectors are never modified, therefore
 *  a single instance can be shared safely between any number of
 *  threads (via std::shared_ptr<const NNTable>). */
class NNTable
{
public:
  /** List of global indices for nearest neighbours for every atom */
    std::vector<std::vector<size_t>>            idx;   // [atom][k] → neighbour index
                                                       //
  /** Periodic image flag for neigherest neighbours.
   *
   * Indicate which periodic image the neighbouring atom belongs to.
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
    std::vector<std::vector<tadah::core::Vec3d>>shift; // [atom][k] → integer (n1,n2,n3)

    NNTable()  = default;
    NNTable(const NNTable&)            = default;
    NNTable& operator=(const NNTable&) = default;

    /* convenience helpers */
    size_t nn_size(size_t i) const              { return idx[i].size(); }
    const std::vector<size_t>& operator[](size_t i) const { return idx[i]; }
};
using NNTablePtr = std::shared_ptr<const NNTable>;
} // namespace tadah::mlip
