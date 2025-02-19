#ifndef TADAH_MLIP_MEMORY_MLIPWORKSPACEMANAGER_H
#define TADAH_MLIP_MEMORY_MLIPWORKSPACEMANAGER_H

#include <tadah/models/memory/ModelsWorkspaceManager.h>
#include <tadah/mlip/memory/IMLIPWorkspaceManager.h>

namespace tadah {
namespace mlip {
namespace memory {

/**
 * @class MLIPWorkspaceManager
 * @brief Implements the IMLIPWorkspaceManager interface to manage workspaces.
 *
 * Manages DesignMatrix workspace, providing methods to obtain and release them.
 */
class MLIPWorkspaceManager : public tadah::models::memory::ModelsWorkspaceManager, public virtual IMLIPWorkspaceManager {

  public:
  /**
   * @brief Constructor.
   */
  MLIPWorkspaceManager();

  /**
   * @brief Destructor.
   *
   * Releases all allocated workspaces.
   */
  ~MLIPWorkspaceManager() override;

  /**
   * @brief Get a DesignMatrix workspace.
   *
   * @param m Number of rows in the matrix.
   * @param n Number of columns in the matrix.
   * @return Pointer to an allocated MLIPWorkspaceManager.
   */
  DesignMatrixWorkspace *getDesignMatrixWorkspace(size_t m, size_t n) override;

  /**
   * @brief Release a DesignMatrix workspace.
   *
   * @param workspace Pointer to the MLIPWorkspaceManager to be released.
   */
  void releaseDesignMatrixWorkspace(DesignMatrixWorkspace *workspace) override;

private:
  DesignMatrixWorkspace *designMatrixWorkspace_; ///< Pointer to the MLIPWorkspaceManager.
};

} // namespace memory
} // namespace mlip
} // namespace tadah

#endif
