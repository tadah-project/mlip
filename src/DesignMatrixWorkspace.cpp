#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tadah/core/lapack.h>
#include <tadah/mlip/memory/DesignMatrixWorkspace.h>

namespace tadah {
namespace mlip {
namespace memory {

DesignMatrixWorkspace::DesignMatrixWorkspace() {}
DesignMatrixWorkspace::~DesignMatrixWorkspace() {}

void DesignMatrixWorkspace::allocate(size_t m_, size_t n_) {
  Phi.resize(m_,n_);
  T.resize(m_);
  Tlabels.resize(m_);
}

bool DesignMatrixWorkspace::isSufficient(size_t m_, size_t n_) const {
  return (Phi.rows() == m_ && Phi.cols() == n_
      && T.size() == m_ && Tlabels.size() == m_);
}

} // namespace memory
} // namespace mlip
} // namespace tadah

