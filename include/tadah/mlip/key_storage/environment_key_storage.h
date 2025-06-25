/**
 * @file EnvironmentKeyStorage.h
 * @brief Implementation of IKeyStorage that reads/writes keys from/to
 * environment variables.
 *
 * By default, environment variables are read-only for user’s global
 * environment. Storing can be done via setenv for the current process if
 * desired.
 */

#ifndef ENVIRONMENTKEYSTORAGE_H
#define ENVIRONMENTKEYSTORAGE_H

#include <tadah/mlip/key_storage/ikey_storage.h>
#include <string>
namespace tadah {
namespace mlip {

/**
 * @class EnvironmentKeyStorage
 * @brief Reads a key from environment variables.
 *
 * Storing is optional: if storing is supported, it uses setenv to update
 * the environment variable only for the current process.
 */
class EnvironmentKeyStorage : public IKeyStorage {
public:
  /**
   * @brief Retrieves the environment variable named @p keyName.
   * @param keyName Name of the env variable.
   * @return The environment variable's value, or empty if not set.
   */
  std::string retrieveKey(const std::string &keyName) override;

  /**
   * @brief Stores a value in the process's environment using setenv.
   * @param keyName Name of the env variable.
   * @param keyValue Value to set.
   * @return true if setenv succeeds, false otherwise.
   */
  bool storeKey(const std::string &keyName,
                const std::string &keyValue) override;
};

} // namespace mlip
} // namespace tadah
#endif // ENVIRONMENTKEYSTORAGE_H
