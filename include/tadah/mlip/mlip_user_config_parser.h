#ifndef TADAH_MLIP_MLIP_user_config_parser_H
#define TADAH_MLIP_MLIP_user_config_parser_H

#include <string>
#include <vector>
#include <tadah/core/iuser_config_parser.h>
#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/**
 * @brief Advanced parser that separates contexturation sections.
 *
 * This parser uses an IUserConfigParser backend (by default,
 * a CoreUserConfigParser) to parse a user contexturation file containing key/value pairs.
 * Then it examines the input (passed as a vector of strings) to detect section delimiters.
 * In our design, a section delimiter is any line starting with "TASK" (case-insensitive).
 *
 * Depending on the merge flag, the MLIP parser may merge the base (un-sectioned)
 * contexturation with every subsequent section (where keys in the section override those in the base).
 *
 * The parser returns a vector of Context objects; the first is the base contexturation,
 * and each additional one is a section (global/task merged) contexturation.
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
   * @brief Parses a user contexturation from a file.
   *
   * Splits the input into sections using "TASK" as a delimiter. If merge is true,
   * then each section is merged with the base contexturation (keys in the section override base).
   *
   * @param filename The contexturation file.
   * @param merge Whether to merge each section with the base contexturation.
   * @return A vector of Context objects representing each section.
   *         The first element is the base contexturation.
   * @throws std::runtime_error if parsing fails.
   */
  std::vector<core::Context> parse(const std::string &filename, bool merge = true) const;

  /**
   * @brief Parses a user contexturation from a vector of strings.
   *
   * @param lines The contexturation lines.
   * @param merge Whether to merge each section with the base contexturation.
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
   * @param lines The raw contexturation lines.
   * @return A vector of sections (each section is a vector of strings).
   */
  std::vector<std::vector<std::string>> splitSections(const std::vector<std::string>& lines) const;

  /**
   * @brief Merges two Context objects.
   *
   * For keys that exist in both, values from the overriding context replace those in base.
   *
   * @param base The base Context.
   * @param override The Context with overriding settings.
   * @return A merged Context.
   */
  core::Context merge(const core::Context &base, const core::Context &override) const;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_MLIP_user_config_parser_H

