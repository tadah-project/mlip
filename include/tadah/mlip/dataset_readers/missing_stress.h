#ifndef TADAH_MLIP_MISSING_STRESS_H
#define TADAH_MLIP_MISSING_STRESS_H

#include <tadah/core/logger.h>

#include <cstddef>
#include <string>

namespace tadah {
namespace mlip {

/**
 * @brief The INFO line of a DFT reader whose file holds structures without a
 * stress tensor.
 *
 * The readers store such a structure with zero stress and stress weight 0
 * (`sweight`), so that a `STRESS true` fit ignores its stress instead of
 * fitting it to zero. Logs nothing when `missing` is 0.
 *
 * @param source  the reader and the file, e.g. "VASP OUTCAR run/OUTCAR".
 * @param missing structures of the file stored without a stress tensor.
 * @param total   structures stored from the file.
 */
inline void log_missing_stress(const std::string &source, std::size_t missing,
                               std::size_t total) {
  if (missing == 0) return;
  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      source + ": " + std::to_string(missing) + " of " +
          std::to_string(total) +
          " structures have no stress tensor; they are stored with zero "
          "stress and stress weight 0, so a STRESS fit ignores them");
}

/**
 * @brief The INFO line of a DFT reader whose file holds structures without
 * forces: they are stored with zero forces and force weight 0 (`fweight`),
 * so that a `FORCE true` fit ignores them. Logs nothing when `missing` is 0.
 */
inline void log_missing_forces(const std::string &source, std::size_t missing,
                               std::size_t total) {
  if (missing == 0) return;
  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      source + ": " + std::to_string(missing) + " of " +
          std::to_string(total) +
          " structures have no forces; they are stored with zero forces "
          "and force weight 0, so a FORCE fit ignores them");
}

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_MISSING_STRESS_H
