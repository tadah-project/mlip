/**
 * @file IKeyStorage.h
 * @brief Abstract interface for retrieving and storing keys (e.g., API tokens).
 */

#ifndef IKEYSTORAGE_H
#define IKEYSTORAGE_H

#include <string>

/**
 * @class IKeyStorage
 * @brief Contract for any key storage mechanism.
 *
 * Classes implementing this interface can store and retrieve keys by name.
 */
class IKeyStorage
{
public:
  virtual ~IKeyStorage() = default;

  /**
   * @brief Retrieves the value associated with @p keyName.
   * @param keyName The name of the key to be retrieved.
   * @return The value if found, otherwise an empty string.
   */
  virtual std::string retrieveKey(const std::string& keyName) = 0;

  /**
   * @brief Stores or updates the value for @p keyName.
   * @param keyName The key name (e.g. "MY_REST_API_KEY").
   * @param keyValue The value to store (e.g. "abc123").
   * @return true if successfully stored, false if not supported or an error occurred.
   */
  virtual bool storeKey(const std::string& keyName, const std::string& keyValue) = 0;
};

#endif // IKEYSTORAGE_H
