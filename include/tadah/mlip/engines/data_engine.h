#ifndef TADAH_MLIP_ENGINES_DATA_ENGINE_H
#define TADAH_MLIP_ENGINES_DATA_ENGINE_H

#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core dataset manipulation functions.
 *
 * Implements conversion, merging, splitting, deduplication, and sampling.
 */
class DataEngine {
public:
  static int convert(const std::vector<std::string>& datasets, const std::string &outfile);
  static int merge(const std::vector<std::string>& datasets, const std::string &outfile);
  static int split(const std::string &dataset, const std::vector<std::string>& outfiles);
  static int dedup(const std::vector<std::string>& datasets, const std::vector<std::string>& outfiles);
  static int sample(const std::vector<std::string>& datasets, const std::string &outfile);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_DATA_ENGINE_H

