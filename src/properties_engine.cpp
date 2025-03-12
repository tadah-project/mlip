#include <tadah/mlip/engines/properties_engine.h>
#include <iostream>
#include <cstdlib>

namespace tadah {
namespace mlip {
namespace engines {

int PropertiesEngine::twobody(const std::string &pot_file, const std::string &outfile,
  const std::vector<double>& range, const std::vector<std::string>& atoms,
  bool force, bool error, int numeric_precision) {
  std::cout << "PropertiesEngine: Computing two-body potential curve (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int PropertiesEngine::ecurve(const std::vector<std::string>& structures, const std::vector<double>& range,
  const std::string &outfile, const std::string &quantity) {
  std::cout << "PropertiesEngine: Computing energy curve for quantity '" << quantity << "' (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int PropertiesEngine::eos(const std::vector<std::string>& structures, const std::vector<double>& range,
  const std::string &outfile) {
  std::cout << "PropertiesEngine: Computing Equation of State (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int PropertiesEngine::defect() {
  std::cout << "PropertiesEngine: Computing defect formation energies (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int PropertiesEngine::surface() {
  std::cout << "PropertiesEngine: Computing surface energies (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int PropertiesEngine::mechanics() {
  std::cout << "PropertiesEngine: Computing bulk modulus and elastic constants (stub)." << std::endl;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

