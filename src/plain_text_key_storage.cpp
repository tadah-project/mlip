/**
 * @file PlainTextKeyStorage.cpp
 * @brief Implementation file for PlainTextKeyStorage class.
 */

#include <tadah/mlip/key_storage/plain_text_key_storage.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

PlainTextKeyStorage::PlainTextKeyStorage()
{
  m_configDir = prepareConfigDirectory();
}

std::string PlainTextKeyStorage::retrieveKey(const std::string& keyName)
{
  return readKeyFromFile(keyName);
}

bool PlainTextKeyStorage::storeKey(const std::string& keyName, const std::string& keyValue)
{
  auto pairs = loadAllKeys();
  bool updated = false;

  // Update existing or append
  for (auto &kv : pairs) {
    if (kv.first == keyName) {
      kv.second = keyValue; // update
      updated = true;
      break;
    }
  }
  if (!updated) {
    pairs.emplace_back(keyName, keyValue);
  }

  return writeAllKeys(pairs);
}

std::string PlainTextKeyStorage::prepareConfigDirectory()
{
  const char* homeDir = std::getenv("HOME");
  if (!homeDir) {
    std::cerr << "Warning: HOME not set; using current directory for keys file.\n";
    return ".";
  }

  std::string configDir = std::string(homeDir) + "/.config/tadah";
  struct stat sb;
  if (stat(configDir.c_str(), &sb) != 0) {
    // Directory does not exist -> mkdir
    if (mkdir(configDir.c_str(), 0700) != 0) {
      std::cerr << "Error: Could not create config directory: " << configDir
                << "\nFalling back to current directory.\n";
      return ".";
    }
  } else if (!S_ISDIR(sb.st_mode)) {
    // Path exists but is not a directory
    std::cerr << "Error: " << configDir << " is not a directory.\n"
              << "Falling back to current directory.\n";
    return ".";
  }

  return configDir;
}

std::string PlainTextKeyStorage::getKeysFilePath() const
{
  return m_configDir + "/keys";
}

std::string PlainTextKeyStorage::readKeyFromFile(const std::string& keyName)
{
  std::ifstream fileIn(getKeysFilePath());
  if (!fileIn.good()) {
    return "";
  }

  std::string line;
  while (std::getline(fileIn, line)) {
    if (line.empty()) continue;

    auto spacePos = line.find(' ');
    if (spacePos == std::string::npos) {
      continue; // skip malformed
    }

    std::string k = line.substr(0, spacePos);
    std::string v = line.substr(spacePos + 1);

    if (k == keyName) {
      return v;
    }
  }

  return "";
}

std::vector<std::pair<std::string, std::string>> PlainTextKeyStorage::loadAllKeys()
{
  std::vector<std::pair<std::string, std::string>> kvPairs;
  std::ifstream fileIn(getKeysFilePath());
  if (!fileIn.good()) {
    return kvPairs;
  }

  std::string line;
  while (std::getline(fileIn, line)) {
    if (line.empty()) continue;
    auto spacePos = line.find(' ');
    if (spacePos == std::string::npos) {
      continue;
    }
    std::string k = line.substr(0, spacePos);
    std::string v = line.substr(spacePos + 1);
    kvPairs.emplace_back(k, v);
  }

  return kvPairs;
}

bool PlainTextKeyStorage::writeAllKeys(const std::vector<std::pair<std::string, std::string>>& kvPairs)
{
  std::ofstream fileOut(getKeysFilePath(), std::ios::trunc);
  if (!fileOut.good()) {
    std::cerr << "Could not write to file: " << getKeysFilePath() << "\n";
    return false;
  }

  for (const auto &kv : kvPairs) {
    fileOut << kv.first << " " << kv.second << "\n";
  }
  return true;
}

