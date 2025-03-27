#ifndef TADAH_MLIP_IENGINE_H
#define TADAH_MLIP_IENGINE_H

#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/**
 * @brief Abstract interface for all engine classes.
 *
 * The engine is a calculator/validator that does not store a Context internally.
 * The same engine instance may be used multiple times with different Context objects.
 */
class IEngine {
public:
  /**
   * @brief Validates whether the given Context has all keys required by this engine.
   * @param ctx A fully prepared Context (merged from CLI or files).
   * @throws std::runtime_error if validation fails.
   */
  virtual void validate(const tadah::core::Context &ctx) = 0;

  /**
   * @brief Executes the engine logic using the provided Context.
   * @param ctx The contextual data with required keys/values.
   * @return An integer status code (0 for success, non-zero for errors).
   * @throws std::runtime_error if runtime errors occur.
   */
  virtual int run(const tadah::core::Context &ctx) = 0;

  /**
   * @brief Virtual destructor for interface compliance.
   */
  virtual ~IEngine() {}
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_IENGINE_H
