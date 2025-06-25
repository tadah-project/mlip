#ifndef TADAH_MLIP_MEMORY_IMLIPWORKSPACEMANAGER_H
#define TADAH_MLIP_MEMORY_IMLIPWORKSPACEMANAGER_H

#include <tadah/models/memory/imodels_workspace_manager.h>
#include <tadah/core/core_types.h>

namespace tadah {
namespace mlip {
namespace memory {

class DesignMatrixWorkspace; ///< Forward declaration of DesignMatrixWorkspace

/**
 * @interface IMLIPWorkspaceManager
 * @brief Interface for managing workspaces specific to the MLIP module.
 *
 * Extends the IModelsWorkspaceManager interface and provides methods for
 * obtaining and releasing DesignMatrix workspace.
 */
class IMLIPWorkspaceManager : public virtual tadah::models::memory::IModelsWorkspaceManager {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~IMLIPWorkspaceManager();

    /**
     * @brief Get a DesignMatrix workspace suitable for the given problem size.
     *
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @param wcopy Keep copy of Phi and T for weighted design matrix and target vector.
     * @return Pointer to an allocated DesignMatrixWorkspace.
     */
    virtual DesignMatrixWorkspace* getDesignMatrixWorkspace(size_t m, size_t n, bool wcopy) = 0;

    /**
     * @brief Release a DesignMatrix workspace.
     *
     * @param workspace Pointer to the DesignMatrixWorkspace to be released.
     */
    virtual void releaseDesignMatrixWorkspace(DesignMatrixWorkspace* workspace) = 0;

};

} // namespace memory
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_MEMORY_IMLIPWORKSPACEMANAGER_H
