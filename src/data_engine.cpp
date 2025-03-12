#include <tadah/mlip/engines/data_engine.h>
#include <iostream>

namespace tadah {
namespace mlip {
namespace engines {

int DataEngine::convert(const std::vector<std::string>& datasets, const std::string &outfile) {
  std::cout << "DataEngine: Converting datasets (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int DataEngine::merge(const std::vector<std::string>& datasets, const std::string &outfile) {
  std::cout << "DataEngine: Merging datasets (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int DataEngine::split(const std::string &dataset, const std::vector<std::string>& outfiles) {
  std::cout << "DataEngine: Splitting dataset (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int DataEngine::dedup(const std::vector<std::string>& datasets, const std::vector<std::string>& outfiles) {
  std::cout << "DataEngine: Deduplicating datasets (stub)." << std::endl;
  return EXIT_SUCCESS;
}

int DataEngine::sample(const std::vector<std::string>& datasets, const std::string &outfile) {
  std::cout << "DataEngine: Sampling dataset (stub)." << std::endl;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

