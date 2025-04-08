#include "tadah/mlip/mlip_user_config_parser.h"
#include "tadah/core/core_user_config_parser.h"
#include "tadah/core/blueprint_parser.h"
#include "tadah/core/logger.h"
#include "tadah/core/utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace tadah {
namespace mlip {

MLIPUserConfigParser::MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend)
  : backendParser_(backend)
{
  // If no backend is provided, create a default parser instance.
  if (!backendParser_) {
    backendParser_ = new tadah::core::CoreUserConfigParser();
  }
}

MLIPUserConfigParser::~MLIPUserConfigParser() {
  // If we created the parser internally (it is a CoreUserConfigParser), then we own it.
  // if (backendParser_ && 
  //     dynamic_cast<const tadah::core::CoreUserConfigParser*>(backendParser_)) {
  //   delete backendParser_;
  // }
}

/**
 * @brief Parses a user config from file, then merges sections if mergeFlag is true.
 */
std::unordered_map<std::string, tadah::core::Context>
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

  return parse(lines, mergeFlag);
}

/**
 * @brief Parses a user config from lines, ensuring:
 *        - The first section is "GLOBAL" if it's non-empty (lines before any TASK).
 *        - Subsequent sections must match an entry in the blueprint’s getTasksMapRef().
 *        - If mergeFlag is true, merges each section with the global context.
 */
std::unordered_map<std::string, tadah::core::Context>
MLIPUserConfigParser::parse(const std::vector<std::string>& lines, bool mergeFlag) const
{
  // Split into named sections: "GLOBAL" + each "TASK...." 
  std::vector<std::pair<std::string, std::vector<std::string>>> namedSections =
      splitSections(lines);

  // The final result: map of <sectionName, Context>
  std::unordered_map<std::string, tadah::core::Context> resultMap;

  // We'll retrieve the blueprint tasks for validation
  const auto &taskMap = tadah::core::BlueprintParser::getTasksMapRef();

  // Parse each section. If it's not "GLOBAL", validate it using blueprint tasks.
  for (std::size_t i = 0; i < namedSections.size(); ++i) {
    const std::string &sectionName = namedSections[i].first;
    const auto &sectionLines       = namedSections[i].second;

    // If sectionName is not "GLOBAL", confirm it exists in the blueprint tasks
    if (sectionName != "GLOBAL") {
      // The blueprint tasks are keyed by the exact string e.g. "TASK.TRAIN", etc.
      // std::cout << "DEBUG: MLIPUserConfigParser::parse: sectionName=" << sectionName << "\n";
      // print all taskMap keys
      // for (const auto &kv : taskMap) {
      //   std::cout << "DEBUG  " << kv.first << "\n";
      // }
      auto found = taskMap.find(sectionName);
      if (found == taskMap.end()) {
        // The user specified a "TASK..." that does not exist in the blueprint
        throw std::runtime_error(
          "MLIPUserConfigParser::parse: Unknown task section \"" + sectionName +
          "\". Must be defined in blueprint tasks."
        );
      }
      // If found, we consider it valid. The backend parser will parse the lines themselves.
    }

    // Parse the lines for this section to produce a Context.  
    tadah::core::Context ctx;
    if (!sectionLines.empty()) {
      ctx = backendParser_->parse(sectionLines);
    }

    // Put it in the map
    resultMap[sectionName] = ctx;
  }

  // If mergeFlag is true, we do the final merges.
  if (mergeFlag) {
    mergeAllSections(resultMap);
  }

  return resultMap;
}

/**
 * @brief Splits lines into named sections:
 *        - The first section is "GLOBAL" if it has lines before "TASK..."
 *        - Each "TASK..." line forms a new name like "TASK.TRAIN" or "TASK.DATA.SUMMARY".
 *
 * Only returns "GLOBAL" if it was non-empty. If no lines for global,
 * we skip it. 
 */
std::vector<std::pair<std::string, std::vector<std::string>>>
MLIPUserConfigParser::splitSections(const std::vector<std::string>& lines) const
{
  // We'll build a list of (sectionName, linesForThisSection)
  std::vector<std::pair<std::string, std::vector<std::string>>> sections;
  sections.reserve(8); // optional

  std::string currentSectionName = "GLOBAL";
  std::vector<std::string> currentSectionLines;

  for (auto &rawLine : lines) {
    // Trim using core::trim(...)
    std::string line = core::trim(rawLine);

    // Skip empty or comment lines
    if (line.empty() || line[0] == '#') {
      continue;
    }

    // Convert to uppercase for checking prefix:
    std::string uppercaseLine = core::to_upper(line);
    // Check if starts with "TASK"
    if (uppercaseLine.size() >= 4 && uppercaseLine.compare(0, 4, "TASK") == 0) {
      // We're about to switch to a new section:
      // finalize the current one if it has lines
      if (!currentSectionLines.empty()) {
        sections.push_back(std::make_pair(currentSectionName, currentSectionLines));
        currentSectionLines.clear();
      }

      // Parse tokens from this line
      std::istringstream iss(line);
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

      // Switch to the new section
      currentSectionName = newSectionName;
    }
    else {
      // This line belongs to the current section (GLOBAL or a previous TASK).
      currentSectionLines.push_back(line);
    }
  }

  // If there's anything left, push it
  if (!currentSectionLines.empty()) {
    sections.push_back(std::make_pair(currentSectionName, currentSectionLines));
  }

  // If the only section is "GLOBAL" but it's empty, remove it
  // so we do not end up with an empty "GLOBAL" in the parse phase.
  if (sections.size() == 1 &&
      sections[0].first == "GLOBAL" &&
      sections[0].second.empty())
  {
    sections.clear();
  }

  return sections;
}

/**
 * @brief Merges all sections in the map with the "GLOBAL" if it exists and is non-empty.
 *
 * We skip merging the global context into itself.
 */
void MLIPUserConfigParser::mergeAllSections(
    std::unordered_map<std::string, tadah::core::Context> &resultMap) const
{
  auto foundGlobal = resultMap.find("GLOBAL");
  if (foundGlobal == resultMap.end()) {
    // No global section present, nothing to merge
    return;
  }

  // Possibly empty global context
  const tadah::core::Context &globalCtx = foundGlobal->second;
  if (globalCtx.size()==0) {
    // If global is empty, merging does nothing
    return;
  }

  // Merge global into each other section
  for (auto &kv : resultMap) {
    // skip the global section itself
    if (kv.first == "GLOBAL") {
      continue;
    }
    // Overwrite keys in kv.second with keys in global
    kv.second = core::Context::merge(globalCtx, kv.second);
  }
}

} // namespace mlip
} // namespace tadah
