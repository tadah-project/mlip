#ifndef TADAH_MLIP_MLIP_USER_CONFIG_PARSER_H
#define TADAH_MLIP_MLIP_USER_CONFIG_PARSER_H

#include <cstddef>
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
 * - Each discovered "TASK ..." line creates a new named section. An in-line
 *   comment ('#' to the end of the line) is not part of the name. If no
 *   tokens follow "TASK", a runtime error is thrown.
 * - A task can exist with zero subsequent key lines (the next line is
 *   another TASK line or the end of the input, or there are only blank and
 *   comment lines before it): its own Context is empty, and merged with the
 *   GLOBAL keys it holds those.
 * - Every TASK section, with or without key lines, must name a blueprint
 *   task, or a runtime error naming it is thrown ('Unknown task section
 *   "BOGUS"'). "TASK global" is no exception: only the lines before the
 *   first TASK line are the GLOBAL section. A section that names a group of
 *   tasks (a blueprint task with tasks of its own, such as "TASK data") is
 *   refused too, naming the group's tasks ('TASK section "DATA" is a group
 *   of tasks, not a task: write 'TASK data <task>', <task> one of: balance,
 *   convert, ...'). Either error is thrown before any section is returned,
 *   so a task file with such a section runs none of its tasks.
 * - A line that gives an option a task retires (the blueprint field
 *   `retired` of the task's key, retired_task_option()) is refused, naming
 *   the line, the task and why (retired_task_option_text()): a line of a
 *   TASK section of that task, or of a file the section includes with
 *   CONFIG ('RESCALE (line 4 of b.task) in TASK data balance: ...'), and a
 *   line of the GLOBAL section, which sets the keys of every task of the
 *   file, when a TASK section of the file retires its key ('RESCALE (line
 *   1 of b.task) in the GLOBAL section reaches TASK data balance ...'),
 *   with the TASK sections of the file that take the key, for which it
 *   belongs in their section. A GLOBAL line of an option that acted only
 *   with another (RetiredKey::only_with: FORCE of tadah data balance acted
 *   only with RESCALE) is accepted: it sets the key for the other tasks.
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
   * @throws std::runtime_error if parsing fails, if a "TASK" line has no
   *         name following it, or if a TASK section names no blueprint task
   *         or a group of tasks.
   */
  std::vector<std::pair<std::string, core::Context>>
  parse(const std::vector<std::string>& lines, bool merge = true) const;

  /**
   * @brief parse(filename, merge) that does not store the key lines
   *        @p skip selects (IUserConfigParser::parse with a KeyLineRule).
   *
   * Each section is read under the rule by the keys of its own lines (the
   * lines above the first TASK line are one file of the rule, each TASK
   * section another; a file a CONFIG line includes is judged by its own
   * lines). @p skipped gets one list per section returned, in the same
   * order: the lines of that section, and of the files it includes, that
   * were not stored. With @p merge, the GLOBAL section's lines come first in
   * the list of each TASK section, as its keys are merged into it.
   */
  std::vector<std::pair<std::string, core::Context>>
  parse(const std::string& filename, bool merge, const core::KeyLineRule& skip,
        std::vector<std::vector<core::SkippedKeyLine>>& skipped) const;

private:
  // const core::IUserConfigParser* backendParser_; ///< The backend parser used for key/value pairs.
  // bool ownsBackend_; std::shared_ptr<core::IUserConfigParser> backendParser_; 
std::shared_ptr<core::IUserConfigParser> backendParser_;

  /**
   * @brief One section of the input: its name, and its lines with the line
   *        number of the first one.
   *
   * `lines` holds every line of the section, blank and comment lines
   * included, so that lines[i] is line first_line + i of the input: the
   * backend parser names the input's line in its errors.
   */
  struct Section {
    std::string name;
    bool task = false;                ///< a TASK section (else GLOBAL)
    std::size_t first_line = 1;       ///< 1-based line number of lines[0]
    std::vector<std::string> lines;
  };

  /**
   * @brief parse(lines) for the lines of @p source (a file name, or empty).
   */
  std::vector<std::pair<std::string, core::Context>>
  parseNamed(const std::vector<std::string>& lines, bool merge,
             const std::string& source,
             const core::KeyLineRule* skip = nullptr,
             std::vector<std::vector<core::SkippedKeyLine>>* skipped =
                 nullptr) const;

  /**
   * @brief Splits lines into named sections based on "TASK" delimiter.
   *
   * If lines exist before the first "TASK", they form a (sectionName="GLOBAL", lines).
   * Each new "TASK some_name" creates a new section named by uppercasing
   * the first token after "TASK", lowercasing the others and joining them
   * with '.' (e.g. "TASK data split" -> "DATA.split"). The tokens are those
   * before an in-line comment ("TASK train   # first" -> "TRAIN"), as for
   * a key line. If there are no tokens following "TASK", a runtime_error
   * is thrown.
   *
   * @param lines The raw config file lines.
   * @return Every TASK section, with or without key lines, in input
   *         order, after a "GLOBAL" section when a line other than a blank
   *         or comment line comes before the first TASK.
   */
  std::vector<Section>
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
