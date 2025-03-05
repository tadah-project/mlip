/**
 * @file KeyStorageFactory.cpp
 * @brief Implementation of the KeyStorageFactory class.
 */

#include <tadah/mlip/key_storage/environment_key_storage.h>
#include <tadah/mlip/key_storage/plain_text_key_storage.h>
#include <tadah/mlip/key_storage/fallback_key_storage.h>
#include <tadah/mlip/key_storage/key_storage_factory.h>
#include <vector>

std::unique_ptr<IKeyStorage> KeyStorageFactory::createEnvThenFile()
{
  // Define storages in order: environment first, then file
  std::vector<std::unique_ptr<IKeyStorage>> storages;
  storages.push_back(std::make_unique<EnvironmentKeyStorage>());
  storages.push_back(std::make_unique<PlainTextKeyStorage>());
  return std::make_unique<FallbackKeyStorage>(std::move(storages));
}

std::unique_ptr<IKeyStorage> KeyStorageFactory::createPlainTextOnly()
{
  return std::make_unique<PlainTextKeyStorage>();
}

std::unique_ptr<IKeyStorage> KeyStorageFactory::createEnvironmentOnly()
{
  return std::make_unique<EnvironmentKeyStorage>();
}

