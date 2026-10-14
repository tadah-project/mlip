#include "tadah/mlip/mlip_user_config_parser.h"
#include "tadah/core/core_user_config_parser.h"
#include "tadah/core/blueprint_parser.h"
#include "tadah/core/logger.h"
#include "tadah/core/utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace tadah {
namespace mlip {

MLIPUserConfigParser::MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend)
  : backendParser_(
      backend
        // if caller supplied a parser, wrap it in a shared_ptr with a no-op deleter
        ? std::shared_ptr<core::IUserConfigParser>(const_cast<core::IUserConfigParser*>(backend),
            [](core::IUserConfigParser*){})
        // otherwise build our own default parser with normal deleter
        : std::make_shared<core::CoreUserConfigParser>()
    )
{}
// MLIPUserConfigParser::~MLIPUserConfigParser() {
//   // Only delete if this object created the backend.
//   if (ownsBackend_) {
//     delete backendParser_;
//   }
//   backendParser_ = nullptr;
//   ownsBackend_    = false;
// }

/**
 * @brief Parses a user config from file, then merges sections if mergeFlag is true.
 */
std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parse(const std::string& filename, bool mergeFlag) const
{
  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    throw std::runtime_error(
      "MLIPUserConfigParser::parse(file): Cannot open file: " + filename
    );
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }

  return parseNamed(lines, mergeFlag, filename);
}

std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parse(
    const std::string& filename, bool mergeFlag, const core::KeyLineRule& skip,
    std::vector<std::vector<core::SkippedKeyLine>>& skipped) const
{
  std::ifstream infile(filename.c_str());
  if (!infile.is_open()) {
    throw std::runtime_error(
      "MLIPUserConfigParser::parse(file): Cannot open file: " + filename
    );
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }

  return parseNamed(lines, mergeFlag, filename, &skip, &skipped);
}

/**
 * @brief Parses a user config from lines, ensuring:
 *        - The first section is "GLOBAL" if it's non-empty (lines before any TASK).
 *        - Every TASK section, with or without key lines, must match an
 *          entry in the blueprint’s getTasksMapRef().
 *        - If mergeFlag is true, merges each section with the global context.
 */
std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parse(const std::vector<std::string>& lines, bool mergeFlag) const
{
  return parseNamed(lines, mergeFlag, "");
}

std::vector<std::pair<std::string, core::Context>>
MLIPUserConfigParser::parseNamed(
    const std::vector<std::string>& lines, bool mergeFlag,
    const std::string& source, const core::KeyLineRule* skip,
    std::vector<std::vector<core::SkippedKeyLine>>* skipped) const
{
  if (skipped) skipped->clear();
  // Split into named sections: "GLOBAL" + each "TASK...." 
  const std::vector<Section> namedSections = splitSections(lines);

  // The final result: vec of <sectionName, Context>
  std::vector<std::pair<std::string, core::Context>> resultVec;

  // We'll retrieve the blueprint tasks for validation
  const auto &taskMap = tadah::core::BlueprintParser::getTasksMapRef();

  // Parse each section. If it's not "GLOBAL", validate it using blueprint tasks.
  for (std::size_t i = 0; i < namedSections.size(); ++i) {
    const std::string &sectionName = namedSections[i].name;
    const auto &sectionLines       = namedSections[i].lines;

    // A TASK section must name a blueprint task, whatever its name ("TASK
    // global" too: only the lines before the first TASK line are GLOBAL).
    if (namedSections[i].task) {
      auto found = taskMap.find(sectionName);
      if (found == taskMap.end()) {
        // The user specified a "TASK..." that does not exist in the blueprint
        throw std::runtime_error(
          "MLIPUserConfigParser::parse: Unknown task section \"" + sectionName +
          "\". Must be defined in blueprint tasks."
        );
      }
      // A group of tasks (a blueprint task with tasks of its own, such as
      // DATA) has no engine: only its tasks run, as TaskFactory requires of a
      // command line. Refused here, before any task of the file runs.
      const std::string prefix = sectionName + ".";
      std::vector<std::string> subtasks;
      for (const auto &kv : taskMap) {
        if (kv.first.compare(0, prefix.size(), prefix) == 0) {
          std::string sub = kv.first.substr(prefix.size());
          std::replace(sub.begin(), sub.end(), '.', ' ');
          subtasks.push_back(sub);
        }
      }
      if (!subtasks.empty()) {
        std::sort(subtasks.begin(), subtasks.end());
        std::string line = core::to_lower(sectionName);
        std::replace(line.begin(), line.end(), '.', ' ');
        std::string list;
        for (const auto &sub : subtasks) list += (list.empty() ? "" : ", ") + sub;
        throw std::runtime_error(
          "MLIPUserConfigParser::parse: TASK section \"" + sectionName +
          "\" is a group of tasks, not a task: write 'TASK " + line +
          " <task>', <task> one of: " + list + ".");
      }
      // A task: the backend parser will parse the lines themselves.
    }

    // Parse the lines for this section to produce a Context. Its errors
    // name the line of the input (and the input's file). Under a rule, the
    // section is one file of it: the keys of its own lines decide.
    core::Context ctx;
    if (skip) {
      skipped->emplace_back();
      ctx = backendParser_->parse(sectionLines, namedSections[i].first_line,
                                  source, *skip, skipped->back());
    } else {
      ctx = backendParser_->parse(sectionLines, namedSections[i].first_line,
                                  source);
    }

    // Put it in the vec
    resultVec.push_back(std::make_pair(sectionName, ctx));
  }

  // If mergeFlag is true, we do the final merges.
  if (mergeFlag) {
    mergeAllSections(resultVec);
    // The lines of GLOBAL not stored belong to every TASK section too, as
    // its keys do.
    if (skipped && !resultVec.empty() && resultVec[0].first == "GLOBAL") {
      const std::vector<core::SkippedKeyLine> global = (*skipped)[0];
      for (std::size_t i = 1; i < skipped->size(); ++i)
        (*skipped)[i].insert((*skipped)[i].begin(), global.begin(),
                             global.end());
    }
  }

  return resultVec;
}

/**
 * @brief Splits lines into named sections:
 *        - The first section is "GLOBAL" if it has key lines before "TASK..."
 *        - Each "TASK..." line forms a new name like "TRAIN" or "DATA.convert",
 *          from the tokens before its in-line comment.
 *
 * Only returns "GLOBAL" if it has a key line. Every TASK section is
 * returned, with or without key lines: its name is checked by the caller,
 * and a keyless task runs on the GLOBAL keys.
 */
std::vector<MLIPUserConfigParser::Section>
MLIPUserConfigParser::splitSections(const std::vector<std::string>& lines) const
{
  std::vector<Section> sections;
  sections.reserve(8); // optional

  // The section being read. Its blank and comment lines are kept, so that
  // its lines keep the input's line numbers. A TASK section is always
  // returned; the GLOBAL section only when it has another line
  // (hasContent).
  Section current{"GLOBAL", false, 1, {}};
  bool hasContent = false;
  auto finish = [&]() {
    if (hasContent || current.task) {
      sections.push_back(std::move(current));
    }
  };

  for (std::size_t n = 0; n < lines.size(); ++n) {
    // Trim using core::trim(...)
    std::string line = core::trim(lines[n]);

    // Empty or comment lines: kept for the numbering only
    if (line.empty() || line[0] == '#') {
      current.lines.push_back(line);
      continue;
    }

    // Convert to uppercase for checking prefix:
    std::string uppercaseLine = core::to_upper(line);
    // Check if starts with "TASK"
    if (uppercaseLine.size() >= 4 && uppercaseLine.compare(0, 4, "TASK") == 0) {
      // We're about to switch to a new section:
      // finalize the current one if it has lines
      finish();

      // Parse tokens from this line, without its in-line comment: the task
      // name is the text before the '#', as the value of a key line is
      // (CoreUserConfigParser drops the comment and the blanks before it).
      std::istringstream iss(core::trim(line.substr(0, line.find('#'))));
      std::vector<std::string> tokens;
      {
        std::string token;
        while (iss >> token) {
          tokens.push_back(token);
        }
      }
      // tokens[0] should be "TASK" (uppercaseLine), we need at least 1 more token
      if (tokens.size() < 2) {
        throw std::runtime_error(
          "MLIPUserConfigParser::splitSections: 'TASK' keyword not followed by any name:\n" + line
        );
      }

      std::string newSectionName;
      newSectionName = core::to_upper(tokens[1]);

      for (std::size_t i = 2; i < tokens.size(); ++i) {
        newSectionName += "." + core::to_lower(tokens[i]);
      }

      // Switch to the new section, which starts on the next line (line n
      // is line n + 1 of the input).
      current = Section{newSectionName, true, n + 2, {}};
      hasContent = false;
    }
    else {
      // This line belongs to the current section (GLOBAL or a previous TASK).
      current.lines.push_back(line);
      hasContent = true;
    }
  }

  // If there's anything left, push it
  finish();

  return sections;
}

/**
 * @brief Merges all sections in the vec with the "GLOBAL" if it exists and is non-empty.
 *
 * We skip merging the global context into itself.
 */
void MLIPUserConfigParser::mergeAllSections(
    std::vector<std::pair<std::string, core::Context>> &resultVec) const
{
  if (resultVec.empty() || resultVec[0].first != "GLOBAL") {
    // No global section present, nothing to merge
    return;
  }

  // Possibly empty global context
  const core::Context &globalCtx = resultVec[0].second;
  if (globalCtx.size()==0) {
    // If global is empty, merging does nothing
    return;
  }

  // Merge global into each other section
  for (auto &pair : resultVec) {
    // skip the global section
    if (pair.first == "GLOBAL") {
      continue;
    }
    // add keys from global to pair.second
    // pair.second takes precedence over global
    pair.second = core::Context::merge(globalCtx, pair.second);
  }
}

} // namespace mlip
} // namespace tadah
