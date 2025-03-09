/**
 * @file FallbackKeyStorage.h
 * @brief An aggregator that tries multiple IKeyStorage instances in order (for reading), 
 *        and writes to a chosen target storage.
 */

#ifndef FALLBACKKEYSTORAGE_H
#define FALLBACKKEYSTORAGE_H

#include <tadah/mlip/key_storage/ikey_storage.h>
#include <vector>
#include <memory>
namespace tadah {
namespace mlip {

/**
 * @class FallbackKeyStorage
 * @brief Aggregates multiple IKeyStorage implementations in a fallback chain.
 *
 * Reading: The first storage that returns a non-empty string is used.  
 * Writing: By default, writes to the last storage in the list. 
 * (This can be adjusted if a different policy is needed.)
 */
class FallbackKeyStorage : public IKeyStorage
{
public:
  /**
   * @brief Constructs with an ordered list of storages to try for retrieval.
   * @param storages A vector of storages, from highest to lowest precedence for reading.
   */
  FallbackKeyStorage(std::vector<std::unique_ptr<IKeyStorage>> storages);

  /**
   * @brief Retrieves the key by checking storages in order, returning the first non-empty result.
   * @param keyName Name of the key to retrieve.
   * @return Key value, or empty string if none of the storages have it.
   */
  std::string retrieveKey(const std::string& keyName) override;

  /**
   * @brief Stores the key in the last storage by default. 
   * @param keyName Name of the key.
   * @param keyValue Value to store.
   * @return true if stored successfully, false otherwise.
   */
  bool storeKey(const std::string& keyName, const std::string& keyValue) override;

private:
  std::vector<std::unique_ptr<IKeyStorage>> m_storages; 
};

} // namespace mlip
} // namespace tadah
#endif // FALLBACKKEYSTORAGE_H

