#ifndef TADAH_MLIP_ENGINES_PROPERTIES_ENGINE_H
#define TADAH_MLIP_ENGINES_PROPERTIES_ENGINE_H

#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides functions for physical property evaluation.
 *
 * Implements functions for computing the two-body potential curve,
 * energy curves (ecurve), EOS, defect formation energies, surface energies,
 * and mechanical properties.
 */
class PropertiesEngine {
public:
  static int twobody(const std::string &pot_file, const std::string &outfile,
                     const std::vector<double>& range, const std::vector<std::string>& atoms,
                     bool force, bool error, int numeric_precision);
  static int ecurve(const std::vector<std::string>& structures, const std::vector<double>& range,
                    const std::string &outfile, const std::string &quantity);
  static int eos(const std::vector<std::string>& structures, const std::vector<double>& range,
                 const std::string &outfile);
  static int defect();
  static int surface();
  static int mechanics();
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_PROPERTIES_ENGINE_H

