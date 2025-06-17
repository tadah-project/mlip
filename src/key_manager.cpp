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
            << "To obtain a key, visit the official website for instructions.\n\n";

  int choice = promptUserChoice();
  switch (choice) {
    case 1:  // Store in context file
    {
      std::cout << "Please enter the key now (Ctrl+C to cancel):\n> ";
      std::getline(std::cin, val);
      if (val.empty()) {
        std::cerr << "No key entered, returning empty.\n";
        return "";
      }
      bool storeOk = storage->storeKey(keyName, val);
      if (!storeOk) {
        std::cerr << "Failed to store key in context.\n";
      } else {
        std::cout << "[Info] Key stored in context file (~/.context/tadah/keys).\n";
      }
      return val;
    }

    case 2:  // Temporarily set environment var for this run
    {
      std::cout << "Please enter the key now (Ctrl+C to cancel):\n> ";
      std::getline(std::cin, val);
      if (val.empty()) {
        std::cerr << "No key entered, returning empty.\n";
        return "";
      }
      setEnvCurrentProcess(keyName, val);
      std::cout << "\n[Info] Key applied to this process only.\n"
                << "It will not be seen by other programs or future runs.\n";
      return val;
    }

    case 3:  // Show instructions for exporting in this shell
    {
      std::cout << "This option does not store or set the key immediately.\n"
                << "It only shows how to export the key in your terminal.\n"
                << "Please copy the example commands below.\n\n";
      // We can guess a random placeholder or let user type it
      // For a real flow, you might also prompt for the key
      std::cout << "Please enter the key (Ctrl+C to cancel):\n> ";
      std::getline(std::cin, val);
      if (val.empty()) {
        std::cerr << "No key entered, returning empty.\n";
        return "";
      }
      showShellExportInstructions(keyName, val);
      return val;
    }

    default: // 4: Abort
      std::cerr << "Aborting. Key remains missing.\n";
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
  std::cout << "Please select an option:\n"
            << "[1] Store the key in a context file (persistent)\n"
            << "[2] Temporarily set environment variable for THIS run only\n"
            << "[3] View instructions for exporting in your shell (useful across multiple runs)\n"
            << "[4] Abort (do nothing)\n"
            << "\nEnter choice [1..4]: ";

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
    std::cerr << "Could not set environment variable in this process.\n";
  }
#endif
}

void KeyManager::showShellExportInstructions(const std::string &keyName, const std::string &keyValue)
{
  std::cout << "------------------------------------------------------------\n"
            << "To export this key for your current TERMINAL session:\n\n"
            << "  export " << keyName << "=\"" << keyValue << "\"\n\n"
            << "After this, any program run in the same terminal will see the key.\n"
            << "------------------------------------------------------------\n"
            << "To make this permanent for ALL future terminal sessions in bash or zsh,\n"
            << "add the same line to your ~/.bashrc or ~/.zshrc file:\n\n"
            << "  echo 'export " << keyName << "=\"" << keyValue << "\"' >> ~/.bashrc\n"
            << "------------------------------------------------------------\n"
            << "Once added, reload your shell or open a new terminal.\n"
            << "Then re-run this program, and it will see that environment variable.\n\n";
}

} // namespace mlip
} // namespace tadah
