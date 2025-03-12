#ifndef TADAH_MLIP_ENGINES_ANALYSIS_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_ENGINE_H

#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides functions for visual analysis.
 *
 * Implements functions for calculating structure descriptors,
 * evaluating basis functions (bfunc), and evaluating cutoff functions.
 */
class AnalysisEngine {
public:
  static int descriptor(const std::vector<std::string>& datasets, const std::string &pot_file,
                        const std::string &outfile, const std::vector<std::string>& indices);
  static int bfunc(const std::string &config_file, const std::string &outfile,
                   const std::vector<double>& range, const std::vector<std::string>& types,
                   bool derivative, bool scale, int numeric_precision);
  static int cutoff(const std::string &outfile, const std::vector<double>& range,
                    const std::vector<std::string>& types, bool derivative, int numeric_precision);
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_ANALYSIS_ENGINE_H

