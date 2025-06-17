/**
 * @file EnvironmentKeyStorage.cpp
 * @brief Implementation of EnvironmentKeyStorage methods.
 */

#include <tadah/mlip/key_storage/environment_key_storage.h>
#include <cstdlib>  // getenv, setenv
#include <iostream>
namespace tadah {
namespace mlip {

std::string EnvironmentKeyStorage::retrieveKey(const std::string& keyName)
{
  if (const char* val = std::getenv(keyName.c_str())) {
    return std::string(val);
  }
  return "";
}

bool EnvironmentKeyStorage::storeKey(const std::string& keyName, const std::string& keyValue)
{
  // setenv updates this process environment only.
  // If persistent environment changes are desired,
  // the user would need to do that manually in shell profiles, etc.
  if (setenv(keyName.c_str(), keyValue.c_str(), 1) == 0) {
    return true;
  }

  std::cerr << "Warning: setenv failed for key: " << keyName << "\n";
  return false;
}

} // namespace mlip
} // namespace tadah
