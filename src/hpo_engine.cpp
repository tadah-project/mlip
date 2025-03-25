#include <tadah/mlip/engines/hpo_engine.h>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int HPOEngine::optimize(tadah::core::Context &context) {
  std::cout << "HPOEngine: Hyperparameter optimization stub executed." << std::endl;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

