/**
 * @file KeyManager.h
 * @brief Provides methods to retrieve and manage API keys for services like (MP, COD, AFLOW), 
 *        offering a 4-option prompt when no key is found.
 */

#ifndef KEYMANAGER_H
#define KEYMANAGER_H

#include <string>
namespace tadah {
namespace mlip {

/**
 * @enum ServiceType
 * @brief Enumerates supported services for which the KeyManager can retrieve API keys.
 */
enum class ServiceType {
  MP,    ///< Materials Project
  COD,   ///< Crystallography Open Database
  AFLOW, ///< AFLOW Project
  // Add more if needed
};

/**
 * @class KeyManager
 * @brief Central utility for retrieving (and if missing, prompting for) service API keys.
 *
 * Uses "EnvThenFile" fallback:
 *   1) Environment variable (e.g. MP_REST_API_KEY)
 *   2) Plain-text config in ~/.config/tadah/keys
 *
 * If still missing, shows four options:
 *   [1] Store key in config file (persistent)
 *   [2] Temporarily set environment variable for this run only
 *   [3] Show how to export it in the shell (so it can be used across multiple runs)
 *   [4] Abort (no key)
 */
class KeyManager {
public:
  /**
   * @brief Retrieves the API key for the specified service.
   *
   * If the key is not found in environment or config, and @p allowPrompt is true,
   * an interactive menu of four options is displayed:
   *   1) Store key in config file (persist for future runs),
   *   2) Temporarily set the environment variable for *this* run only,
   *   3) Show instructions on how to export for the shell (useful across runs),
   *   4) Abort (do nothing, return empty).
   *
   * @param service      The target service (MP, COD, AFLOW).
   * @param allowPrompt  If false, no prompts occur; the method returns empty if key is missing.
   * @return The acquired key, or an empty string if user aborts or no key provided.
   */
  static std::string getServiceKey(ServiceType service, bool allowPrompt = true);

private:
  /// Maps the enum to the environment variable name (e.g. "MP_REST_API_KEY").
  static std::string resolveKeyName(ServiceType service);

  /// A friendly string for prompts/logs (e.g. "Materials Project").
  static std::string getServiceFriendlyName(ServiceType service);

  /// Shows the four-option menu and returns the user's choice (1..4).
  static int promptUserChoice();

  /**
   * @brief Sets the environment variable in the current process only,
   *        so it is visible to subsequent code in this process but not in the user's shell.
   *
   * @param keyName  The environment variable name (e.g. "MP_REST_API_KEY").
   * @param keyValue The value to set.
   */
  static void setEnvCurrentProcess(const std::string& keyName, const std::string& keyValue);

  /**
   * @brief Shows instructions for exporting in the user’s shell so the 
   *        variable persists across multiple runs *in that same terminal session.*
   *
   * Also hints how to place it in shell configuration (e.g. ~/.bashrc) if desired.
   *
   * @param keyName  The env variable name.
   * @param keyValue The key value to be exported.
   */
  static void showShellExportInstructions(const std::string& keyName, const std::string& keyValue);
};
} // namespace mlip
} // namespace tadah
#endif // KEYMANAGER_H
