#include <tadah/mlip/parallel.h>

#include <utility>  // std::move

namespace tadah {
namespace mlip {

namespace {
/* Process-wide parallel backend. Empty by default (=> parallel_for uses the
 * historical OpenMP loop). Set by the application layer (HPO) for each run,
 * to route MLIP kernels onto the run's ScriptExecutor pool, and cleared when
 * the run ends. Not guarded by a mutex: it is installed and cleared only
 * while no parallel_for runs, never concurrently with kernels. */
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
