/**
 * @file FallbackKeyStorage.cpp
 * @brief Implementation of the aggregator for multiple storages in a fallback chain.
 */

#include <tadah/mlip/key_storage/fallback_key_storage.h>
namespace tadah {
namespace mlip {

FallbackKeyStorage::FallbackKeyStorage(std::vector<std::unique_ptr<IKeyStorage>> storages)
  : m_storages(std::move(storages))
{
}

std::string FallbackKeyStorage::retrieveKey(const std::string& keyName)
{
  // Check each storage in order, return the first non-empty result
  for (auto &storage : m_storages) {
    std::string val = storage->retrieveKey(keyName);
    if (!val.empty()) {
      return val;
    }
  }
  return ""; // Not found in any storage
}

bool FallbackKeyStorage::storeKey(const std::string& keyName, const std::string& keyValue)
{
  if (m_storages.empty()) {
    return false;
  }
  // For simplicity: store in the last storage
  return m_storages.back()->storeKey(keyName, keyValue);
}

std::string FallbackKeyStorage::location() const
{
  return m_storages.empty() ? std::string("no storage")
                            : m_storages.back()->location();
}

} // namespace mlip
} // namespace tadah
