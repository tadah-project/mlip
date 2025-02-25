#include <tadah/mlip/memory/MLIPWorkspaceManager.h>
#include <tadah/mlip/memory/DesignMatrixWorkspace.h>

namespace tadah {
namespace mlip {
namespace memory {

MLIPWorkspaceManager::MLIPWorkspaceManager()
    : designMatrixWorkspace_(nullptr) {}

MLIPWorkspaceManager::~MLIPWorkspaceManager() {
    releaseDesignMatrixWorkspace(designMatrixWorkspace_);
}

DesignMatrixWorkspace* MLIPWorkspaceManager::getDesignMatrixWorkspace(size_t m, size_t n, bool wcopy) {
    if (!designMatrixWorkspace_ || !designMatrixWorkspace_->isSufficient(m, n)) {
        releaseDesignMatrixWorkspace(designMatrixWorkspace_);
        designMatrixWorkspace_ = new DesignMatrixWorkspace(wcopy);
        designMatrixWorkspace_->allocate(m, n);
    }
    return designMatrixWorkspace_;
}

void MLIPWorkspaceManager::releaseDesignMatrixWorkspace(DesignMatrixWorkspace* workspace) {
    if (workspace) {
        delete workspace;
        workspace = nullptr;
    }
    designMatrixWorkspace_ = nullptr;
}


} // namespace memory
} // namespace mlip
} // namespace tadah
