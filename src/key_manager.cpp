/**
 * @file KeyManager.cpp
 * @brief Implements a KeyManager that provides a four-option prompt 
 *        if a requested key is missing.
 */
#include <tadah/mlip/key_storage/key_manager.h>
#include <tadah/mlip/key_storage/key_storage_factory.h>

#include <iostream>
#include <string>
#include <cstdlib>   // setenv on Unix-likes
namespace tadah {
namespace mlip {

std::string KeyManager::getServiceKey(ServiceType service, bool allowPrompt)
{
  // 1) Resolve which env/context key to look for
  const std::string keyName = resolveKeyName(service);

  // 2) Acquire fallback storage that checks environment first, then the file
  auto storage = KeyStorageFactory::createEnvThenFile();

  // 3) Attempt to retrieve the key
  std::string val = storage->retrieveKey(keyName);
  if (!val.empty() || !allowPrompt) {
    // Key found, or prompting disallowed => just return
    return val;
  }

  // Key is missing and we can prompt -> show the 4-option menu
  std::cout << "\nNo key found for " << getServiceFriendlyName(service)
            << " (" << keyName << ").\n"
            << "The website of the service says how a key is obtained.\n\n";

  int choice = promptUserChoice();
  switch (choice) {
    case 1:  // Store in the keys file
    {
      std::cout << "The key (Ctrl+C cancels):\n> ";
      std::getline(std::cin, val);
      if (val.empty()) {
        std::cerr << "No key entered; the key stays missing.\n";
        return "";
      }
      bool storeOk = storage->storeKey(keyName, val);
      if (!storeOk) {
        std::cerr << "The key cannot be written to " << storage->location()
                  << "; it is used for this run only.\n";
      } else {
        std::cout << "The key is stored in " << storage->location() << ".\n";
      }
      return val;
    }

    case 2:  // Temporarily set environment var for this run
    {
      std::cout << "The key (Ctrl+C cancels):\n> ";
      std::getline(std::cin, val);
      if (val.empty()) {
        std::cerr << "No key entered; the key stays missing.\n";
        return "";
      }
      setEnvCurrentProcess(keyName, val);
      std::cout << "\nThe key is set for this run only; other programs and "
                   "later runs do not see it.\n";
      return val;
    }

    case 3:  // Show instructions for exporting in this shell
    {
      std::cout << "This option neither stores nor sets the key: it shows "
                   "the commands that export it in a shell.\n\n";
      // The commands are shown with the key itself.
      std::cout << "The key (Ctrl+C cancels):\n> ";
      std::getline(std::cin, val);
      if (val.empty()) {
        std::cerr << "No key entered; the key stays missing.\n";
        return "";
      }
      showShellExportInstructions(keyName, val);
      return val;
    }

    default: // 4: Abort
      std::cerr << "Aborted; the key stays missing.\n";
      return "";
  }
}

std::string KeyManager::resolveKeyName(ServiceType service)
{
  switch (service) {
    case ServiceType::MP:    return "MP_REST_API_KEY";
    case ServiceType::COD:   return "COD_REST_API_KEY";
    case ServiceType::AFLOW: return "AFLOW_REST_API_KEY";
  }
  return "UNKNOWN_SERVICE_KEY";
}

std::string KeyManager::getServiceFriendlyName(ServiceType service)
{
  switch (service) {
    case ServiceType::MP:    return "Materials Project";
    case ServiceType::COD:   return "Crystallography Open Database (COD)";
    case ServiceType::AFLOW: return "AFLOW Project";
  }
  return "Unknown Service";
}

int KeyManager::promptUserChoice()
{
  // Four options now
  std::cout << "Options:\n"
            << "[1] store the key in the keys file (kept for later runs)\n"
            << "[2] set the environment variable for this run only\n"
            << "[3] show the commands that export it in a shell (for several runs)\n"
            << "[4] abort (nothing is done)\n"
            << "\nChoice [1..4]: ";

  std::string line;
  std::getline(std::cin, line);
  if (line.empty()) {
    return 4; // default to abort if user just hits Enter
  }
  char c = line[0];
  if (c == '1') return 1;
  if (c == '2') return 2;
  if (c == '3') return 3;
  return 4; // invalid or '4' => abort
}

void KeyManager::setEnvCurrentProcess(const std::string &keyName, const std::string &keyValue)
{
#if defined(_WIN32)
  // Windows example: _putenv_s
  // For demonstration; might need <cstdlib> or <corecrt.h>
  // _putenv_s(keyName.c_str(), keyValue.c_str());
  // or use SetEnvironmentVariable (WinAPI call)
  std::cout << "[Warning] Setting environment variables at runtime on Windows "
               "might require different API calls.\n";
#else
  // Unix-like approach
  if (setenv(keyName.c_str(), keyValue.c_str(), 1) != 0) {
    std::cerr << "The environment variable " << keyName
              << " cannot be set in this run.\n";
  }
#endif
}

void KeyManager::showShellExportInstructions(const std::string &keyName, const std::string &keyValue)
{
  std::cout << "------------------------------------------------------------\n"
            << "The key is exported for this terminal session by:\n\n"
            << "  export " << keyName << "=\"" << keyValue << "\"\n\n"
            << "Programs run later in this terminal see it.\n"
            << "------------------------------------------------------------\n"
            << "A line in ~/.bashrc or ~/.zshrc exports it in every shell, "
               "such as the one this command adds:\n\n"
            << "  echo 'export " << keyName << "=\"" << keyValue << "\"' >> ~/.bashrc\n"
            << "------------------------------------------------------------\n"
            << "A new terminal, or a reload of the shell, reads the line; a "
               "later run of tadah then sees the variable.\n\n";
}

} // namespace mlip
} // namespace tadah
