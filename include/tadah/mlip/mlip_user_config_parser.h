#ifndef TADAH_MLIP_MLIP_USER_CONFIG_PARSER_H
#define TADAH_MLIP_MLIP_USER_CONFIG_PARSER_H

#include <string>
#include <vector>
#include <tadah/core/iuser_config_parser.h>
#include <tadah/core/context.h>
#include <memory>

namespace tadah {
namespace mlip {

/**
 * @brief Advanced parser for user config that separates tasks sections.
 *
 * This parser uses an IUserConfigParser backend (by default,
 * a CoreUserConfigParser) to parse a user config file containing 
 * key/value pairs. It then analyzes the input lines to detect section
 * delimiters. In this design, any line starting with "TASK" (case-insensitive)
 * is treated as the start of a new section with a name derived from the 
 * text following "TASK".
 *
 * - If lines appear before the first "TASK", they form a "GLOBAL" section
 *   (returned only if that section has at least one key-value line).
 * - Each discovered "TASK ..." line creates a new named section. If no tokens
 *   follow "TASK", a runtime error is thrown.
 * - A task can exist with zero subsequent lines, meaning an empty Context.
 *
 * Depending on the merge flag, each task section can optionally be merged 
 * with the global context, if a global context exists, where task keys 
 * override global ones.
 *
 * @note The parser returns a vec of <section-name, Context>.
 *       "GLOBAL" is only included if it is non-empty. The user may have 
 *       any number of tasks, each with or without lines.
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
   *                If none provided, a default instance is created 
   *                internally (and later destroyed), otherwise ownership
   *                remains with the caller.
   */
  explicit MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend = nullptr);

  /**
   * @brief Destructor.
   *
   * If constructed with a default backend, it is destroyed here. Otherwise 
   * no action is taken.
   */
  // ~MLIPUserConfigParser();

  /**
   * @brief Parses a user config from a file.
   *
   * Splits the input into sections using "TASK" as a delimiter. If merge is true,
   * each section is merged with a global Context (if it exists).
   *
   * @param filename The config file path to parse.
   * @param merge Whether to merge each section with the global Context.
   * @return A vec of <section-name, Context> for each section discovered. 
   *         "GLOBAL" is included only if non-empty.
   * @throws std::runtime_error if parsing fails or file cannot be opened.
   */
  std::vector<std::pair<std::string, core::Context>>
  parse(const std::string &filename, bool merge = true) const;

  /**
   * @brief Parses a user config from a vector of strings.
   *
   * @param lines The lines of the config file.
   * @param merge Whether to merge each section with the global Context.
   * @return A vec of <section-name, Context> for each section discovered.
   *         "GLOBAL" is included only if non-empty.
   * @throws std::runtime_error if parsing fails, or if a "TASK" 
   *         line has no name following it.
   */
  std::vector<std::pair<std::string, core::Context>>
  parse(const std::vector<std::string>& lines, bool merge = true) const;

private:
  // const core::IUserConfigParser* backendParser_; ///< The backend parser used for key/value pairs.
  // bool ownsBackend_; std::shared_ptr<core::IUserConfigParser> backendParser_; 
std::shared_ptr<core::IUserConfigParser> backendParser_;
  /**
   * @brief Splits lines into named sections based on "TASK" delimiter.
   *
   * If lines exist before the first "TASK", they form a (sectionName="GLOBAL", lines).
   * Each new "TASK some_name" creates a new section named by uppercasing 
   * subsequent tokens and joining them with '.' (e.g. "TASK data split" -> 
   * "DATA.SPLIT"). If there are no tokens following "TASK", a runtime_error 
   * is thrown.
   *
   * @param lines The raw config file lines.
   * @return A vector of (sectionName, vector-of-lines) for each section.
   *         Will not return a "GLOBAL" pair if no lines existed before 
   *         the first TASK (i.e. truly empty).
   */
  std::vector<std::pair<std::string, std::vector<std::string>>>
  splitSections(const std::vector<std::string>& lines) const;


  /**
   * @brief Helper that merges each non-empty section with the global context 
   *        if 'GLOBAL' is found and not empty.
   *
   * @param resultVec A vec of <sectionName, Context>.
   *                  On completion, every non-GLOBAL context is merged 
   *                  with the global context (if present and non-empty).
   */
  void mergeAllSections(std::vector<std::pair<std::string, core::Context>> &resultVec) const;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_MLIP_USER_CONFIG_PARSER_H
