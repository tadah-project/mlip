#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/core/config.h>
#include <tadah/mlip/structure_db.h>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int PredictEngine::predict(const tadah::core::Config &pot_config, const tadah::mlip::StructureDB &stdb) {
  std::cout << "PredictEngine: Performing prediction (stub implementation)." << std::endl;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

