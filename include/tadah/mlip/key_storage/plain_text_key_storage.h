/**
 * @file PlainTextKeyStorage.h
 * @brief Implementation of IKeyStorage that reads/writes plain-text keys in ~/.context/tadah/keys.
 */

#ifndef PLAINTEXTKEYSTORAGE_H
#define PLAINTEXTKEYSTORAGE_H

#include <tadah/mlip/key_storage/ikey_storage.h>
#include <vector>
namespace tadah {
namespace mlip {

/**
 * @class PlainTextKeyStorage
 * @brief Handles storage of keys in a plain-text file.
 *
 * The file is located in "~/.context/tadah/keys" if possible, otherwise "." 
 * is used as a fallback.
 *
 * Format of each line: KEY_NAME value
 * Example: MY_REST_API_KEY abc123
 *
 * No encryption is performed in this implementation.
 */
class PlainTextKeyStorage : public IKeyStorage
{
public:
  /**
   * @brief Constructor ensures the directory is set up.
   */
  PlainTextKeyStorage();

  /**
   * @brief Retrieves @p keyName from the keys file (does not check environment).
   * @param keyName The name of the key to look up.
   * @return The value if found, otherwise empty.
   */
  std::string retrieveKey(const std::string& keyName) override;

  /**
   * @brief Writes or updates a line in the keys file, ignoring environment variables.
   * @param keyName The name of the key.
   * @param keyValue The value to associate with the key.
   * @return true on successful file write, false on error.
   */
  bool storeKey(const std::string& keyName, const std::string& keyValue) override;

private:
  std::string m_contextDir; ///< Directory path used for storing the "keys" file.

  std::string prepareContextDirectory();
  std::string getKeysFilePath() const;

  // Helpers for file read/write
  std::string readKeyFromFile(const std::string& keyName);
  std::vector<std::pair<std::string, std::string>> loadAllKeys();
  bool writeAllKeys(const std::vector<std::pair<std::string, std::string>>& kvPairs);
};

} // namespace mlip
} // namespace tadah
#endif // PLAINTEXTKEYSTORAGE_H

