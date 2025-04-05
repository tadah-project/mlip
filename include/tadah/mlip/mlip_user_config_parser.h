#ifndef TADAH_MLIP_MLIP_user_config_parser_H
#define TADAH_MLIP_MLIP_user_config_parser_H

#include <string>
#include <vector>
#include <unordered_map>
#include <tadah/core/iuser_config_parser.h>
#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/**
 * @brief Advanced parser for user config that separates tasks sections.
 *
 * This parser uses an IUserConfigParser backend (by default,
 * a CoreUserConfigParser) to parse a user config file containing key/value pairs.
 * Then it examines the input (passed as a vector of strings) to detect section delimiters.
 * In this design, a section delimiter is any line starting with "TASK" (case-insensitive).
 *
 * For example, a config file may look like:
 * ```
 * # This is a comment
 * # Global settings
 * RCUT2B  5.0
 * TASK data split
 * # Data split settings
 * DBFILE data.db
 * ...
 * TASK train model
 * # Training settings
 * MODEL M_KRR Kern_Linear
 * ...
 * ```
 *
 * Depending on the merge flag, the MLIP parser may merge the global configuration
 * with every subsequent section (where keys in the section override those in the global).
 *
 * The parser returns an unordered_map of Context objects.
 * The global section key is GLOBAL, and the task sections are named after the
 * task name by capitalizing words after "TASK" keyword and concatenating with ".".
 * (e.g., TASK data split -> DATA.SPLIT).
 */
class MLIPUserConfigParser {
public:
  /**
   * @brief Constructs an MLIPUserConfigParser.
   *
   * Optionally, a different backend may be provided (if supporting different
   * input formats in the future). The default is to use a CoreUserConfigParser.
   *
   * @param backend Pointer to a backend IUserConfigParser.
   *                Ownership is not transferred; if none provided, a default instance is used.
   */
  explicit MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend = nullptr);

  /**
   * @brief Destructor.
   *
   * Cleans up the backend parser if it was created internally.
   */
  ~MLIPUserConfigParser();

  /**
   * @brief Parses a user config from a file.
   *
   * Splits the input into sections using "TASK" as a delimiter. If merge is true,
   * then each section is merged with the global Context (keys in the section override global).
   *
   * @param filename The config file.
   * @param merge Whether to merge each section with the global Context.
   * @return A map of Context objects representing each section.
   * @throws std::runtime_error if parsing fails.
   */
  std::unordered_map<std::string,core::Context> parse(const std::string &filename, bool merge = true) const;

  /**
   * @brief Parses a user config from a vector of strings.
   *
   * @param lines The config file lines.
   * @param merge Whether to merge each section with the global Context.
   * @return A map of Context objects representing each section.
   * @throws std::runtime_error if parsing fails.
   */
  std::unordered_map<std::string, core::Context> parse(const std::vector<std::string>& lines, bool merge = true) const;

private:
  const core::IUserConfigParser* backendParser_; ///< The backend parser used to read key/value pairs.

  /**
   * @brief Splits lines into sections based on delimiter.
   *
   * Returns a vector of vector<string>, each representing a section.
   * Lines that start with "TASK" (case-insensitive) are treated as delimiters.
   *
   * @param lines The raw config file lines.
   * @return A vector of sections (each section is a vector of strings).
   */
  std::vector<std::vector<std::string>> splitSections(const std::vector<std::string>& lines) const;

  /**
   * @brief Merges two Context objects.
   *
   * For keys that exist in both, values from the task context replace those in global.
   *
   * @param global The global Context.
   * @param task The Context with task settings.
   * @return A merged Context.
   */
  core::Context merge(const core::Context &global, const core::Context &task) const;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_MLIP_user_config_parser_H

