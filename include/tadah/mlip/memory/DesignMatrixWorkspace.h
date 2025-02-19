#ifndef TADAH_MLIP_MEMORY_DESIGNMATRIXWORKSPACE_H
#define TADAH_MLIP_MEMORY_DESIGNMATRIXWORKSPACE_H

#include <tadah/core/core_types.h>

namespace tadah {
namespace mlip {
namespace memory {

/**
 * @class DesignMatrixWorkspace
 * @brief Manages memory allocations for DesignMatrix computations.
 *
 * The DesigmMatrixWorkspace class handles the memory required for various DM
 * algorithms. It ensures that memory is allocated and deallocated appropriately
 * and provides storage for intermediate computations.
 */
class DesignMatrixWorkspace {
public:
  /**
   * @brief Constructor.
   */
  DesignMatrixWorkspace();

  /**
   * @brief Destructor.
   *
   * Releases all allocated memory.
   */
  ~DesignMatrixWorkspace();

  /**
   * @brief Allocate workspace for the specified problem size and algorithm.
   *
   * @param m Number of rows in the matrix.
   * @param n Number of columns in the matrix.
   */
  void allocate(size_t m_, size_t n_);

  /**
   * @brief Check if the workspace is sufficient for the given problem size and
   * algorithm.
   *
   * @param m Number of rows in the matrix.
   * @param n Number of columns in the matrix.
   * @return True if the workspace is sufficient, false otherwise.
   */
  bool isSufficient(size_t m_, size_t n_) const;

  phi_type Phi;
  t_type T;
  t_type Tlabels; // 0-Energy, 1-Force, 2-Stress  // TODO should be array of int not double


private:
  // Disable copying
  DesignMatrixWorkspace(const DesignMatrixWorkspace &) = delete;
  DesignMatrixWorkspace &operator=(const DesignMatrixWorkspace &) = delete;
};

} // namespace memory
} // namespace mlip
} // namespace tadah

#endif
