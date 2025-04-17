#include <tadah/mlip/engines/explain_engine.h>
#include <tadah/core/context.h>
#include <iostream>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

namespace tadah {
namespace mlip {
namespace engines {

int ExplainEngine::run(mlip::Task &task) {
  int rank = 0;
  #ifdef TADAH_BUILD_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  #endif
  if (rank!=0) return EXIT_SUCCESS;

  const auto &taskContext = task.getContext();
  std::cout << "ExplainEngine: " << taskContext << std::endl;


  return EXIT_SUCCESS;
}
void ExplainEngine::validate(const tadah::mlip::Task &task) {
  const auto &context = task.getContext();
  if (!context.contains("OPTION")) {
    throw std::runtime_error("ExplainEngine requires OPTION parameter to be set");
  }
}

} // namespace engines
} // namespace mlip
} // namespace tadah

