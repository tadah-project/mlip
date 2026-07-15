#include <tadah/mlip/parallel.h>

#include <utility>  // std::move

namespace tadah {
namespace mlip {

namespace {
/* Process-wide parallel backend. Empty by default (=> parallel_for uses the
 * historical OpenMP loop). Set once at startup by the application layer (HPO)
 * to route MLIP kernels onto its shared ScriptExecutor pool. Not guarded by a
 * mutex: it is installed before any parallel_for runs and never swapped
 * concurrently with kernels. */
ParallelBackend g_backend;
}  // namespace

void set_parallel_backend(ParallelBackend backend) {
  g_backend = std::move(backend);
}

bool has_parallel_backend() {
  return static_cast<bool>(g_backend);
}

const ParallelBackend& detail_parallel_backend() {
  return g_backend;
}

}  // namespace mlip
}  // namespace tadah
