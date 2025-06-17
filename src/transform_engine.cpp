#include <tadah/mlip/engines/transform_engine.h>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int TransformEngine::rescale(const std::vector<std::string>& datasets, double eshift, bool scale, int numeric_precision) {
  std::cout << "TransformEngine: Rescaling datasets (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int TransformEngine::rotate(const std::vector<std::string>& datasets, const std::string &axis, double angle, const std::string &outfile) {
  std::cout << "TransformEngine: Rotating structures (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int TransformEngine::shear() {
  std::cout << "TransformEngine: Shearing structures (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int TransformEngine::changeCoords() {
  std::cout << "TransformEngine: Changing coordinate systems (stub)." << std::endl;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

