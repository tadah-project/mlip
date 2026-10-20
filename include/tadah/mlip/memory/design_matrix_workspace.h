#ifndef TADAH_MLIP_MEMORY_DESIGNMATRIXWORKSPACE_H
#define TADAH_MLIP_MEMORY_DESIGNMATRIXWORKSPACE_H

#include <tadah/core/core_types.h>

#include <cstdint>

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
  DesignMatrixWorkspace(bool wcopy);

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
   * @param wcopy Whether the copy of Phi and T (Phi_cpy, T_cpy) is needed:
   *        a workspace made without it is not sufficient then.
   * @return True if the workspace is sufficient, false otherwise.
   */
  bool isSufficient(size_t m_, size_t n_, bool wcopy = false) const;

  tadah::core::phi_type Phi, Phi_cpy; // Design matrix and space for its copy
  tadah::core::t_type T, T_cpy; // Target vector and space for its copy
  tadah::core::t_type Tlabels; // 0-Energy, 1-Force, 2-Stress  // TODO should be array of int not double
  tadah::core::t_type wfactors; // Weighting factors for Phi

  /** The stamp of the build whose rows the workspace holds: each
   *  DesignMatrix::build takes a new one (next_build_stamp()) and keeps it,
   *  so a design matrix tells its own rows from those of another design
   *  matrix built in the same workspace since (DesignMatrix::balance). */
  std::uint64_t build_stamp = 0;
  /** A stamp no build of the process has had (from 1, increasing). */
  static std::uint64_t next_build_stamp();

private:
  // Disable copying
  bool wcopy; // Keep copy of Phi and T for weighted design matrix and target vector
  DesignMatrixWorkspace(const DesignMatrixWorkspace &) = delete;
  DesignMatrixWorkspace &operator=(const DesignMatrixWorkspace &) = delete;
};

} // namespace memory
} // namespace mlip
} // namespace tadah

#endif
