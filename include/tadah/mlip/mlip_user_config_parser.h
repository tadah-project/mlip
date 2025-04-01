#ifndef TADAH_MLIP_MLIP_user_config_parser_H
#define TADAH_MLIP_MLIP_user_config_parser_H

#include <string>
#include <vector>
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
 * In our design, a section delimiter is any line starting with "TASK" (case-insensitive).
 *
 * Depending on the merge flag, the MLIP parser may merge the base configuration
 * with every subsequent section (where keys in the section override those in the base).
 *
 * The parser returns a vector of Context objects; the first is the base configuration,
 * and each additional one is a section (base/task merged) configuration.
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
   * then each section is merged with the base Context (keys in the section override base).
   *
   * @param filename The config file.
   * @param merge Whether to merge each section with the base Context.
   * @return A vector of Context objects representing each section.
   *         The first element is the base Context.
   * @throws std::runtime_error if parsing fails.
   */
  std::vector<core::Context> parse(const std::string &filename, bool merge = true) const;

  /**
   * @brief Parses a user config from a vector of strings.
   *
   * @param lines The config file lines.
   * @param merge Whether to merge each section with the base Context.
   * @return A vector of Context objects representing each section.
   * @throws std::runtime_error if parsing fails.
   */
  std::vector<core::Context> parse(const std::vector<std::string>& lines, bool merge = true) const;

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
   * For keys that exist in both, values from the task context replace those in base.
   *
   * @param base The base Context.
   * @param task The Context with task settings.
   * @return A merged Context.
   */
  core::Context merge(const core::Context &base, const core::Context &task) const;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_MLIP_user_config_parser_H

