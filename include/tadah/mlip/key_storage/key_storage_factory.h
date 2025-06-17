/**
 * @file KeyStorageFactory.h
 * @brief Factory for constructing complex IKeyStorage objects or single backends.
 */

#ifndef KEYSTORAGEFACTORY_H
#define KEYSTORAGEFACTORY_H

#include <memory>
#include <string>
#include <tadah/mlip/key_storage/ikey_storage.h>
namespace tadah {
namespace mlip {

/**
 * @class KeyStorageFactory
 * @brief Produces aggregated or single IKeyStorage instances.
 */
class KeyStorageFactory
{
public:
  /**
   * @brief Creates a fallback storage that checks environment first, then file, etc.
   * @return A unique_ptr to IKeyStorage implementing the fallback strategy.
   */
  static std::unique_ptr<IKeyStorage> createEnvThenFile();

  /**
   * @brief Creates a plain-text-only storage instance (no environment checks).
   * @return A unique_ptr to IKeyStorage that uses plain-text only.
   */
  static std::unique_ptr<IKeyStorage> createPlainTextOnly();

  /**
   * @brief Creates an environment-only storage instance (no file fallback).
   * @return A unique_ptr to IKeyStorage that uses environment only.
   */
  static std::unique_ptr<IKeyStorage> createEnvironmentOnly();
};

} // namespace mlip
} // namespace tadah
#endif // KEYSTORAGEFACTORY_H

