#ifndef TADAH_MLIP_ENGINES_TRANSFORM_ENGINE_H
#define TADAH_MLIP_ENGINES_TRANSFORM_ENGINE_H

#include <string>
#include <vector>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Provides core structure transformation functions.
 *
 * Implements rescaling, rotation, shearing, and coordinate conversion.
 */
class TransformEngine {
public:
  static int rescale(const std::vector<std::string>& datasets, double eshift, bool scale, int numeric_precision);
  static int rotate(const std::vector<std::string>& datasets, const std::string &axis, double angle, const std::string &outfile);
  static int shear();
  static int changeCoords();
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_TRANSFORM_ENGINE_H

