#include "tadah/mlip/mlip_user_config_parser.h"
#include "tadah/core/core_user_config_parser.h"
#include "tadah/core/blueprint_parser.h"
#include "tadah/core/logger.h"
#include "tadah/core/utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace tadah {
namespace mlip {

MLIPUserConfigParser::MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend)
  : backendParser_(backend)
{
  // If no backend is provided, use the default CoreUserConfigParser.
  if (!backendParser_) {
    backendParser_ = new tadah::core::CoreUserConfigParser();
  }
}

MLIPUserConfigParser::~MLIPUserConfigParser() {
  if (backendParser_) {
    delete backendParser_;
  }
}

std::vector<tadah::core::Context> MLIPUserConfigParser::parse(const std::string &filename, bool mergeFlag) const {
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("MLIPUserConfigParser::parse: Cannot open file: " + filename);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }
  return parse(lines, mergeFlag);
}

std::vector<tadah::core::Context> MLIPUserConfigParser::parse(const std::vector<std::string>& lines, bool mergeFlag) const {
  // First, split the lines into sections based on "TASK" delimiter.
  std::vector<std::vector<std::string>> sections = splitSections(lines);
  std::vector<tadah::core::Context> contexts;
  // For each section, use the backend parser to produce a Context.
  // The first section is considered the base Context.
  for (const auto &secLines : sections) {
    tadah::core::Context ctx = backendParser_->parse(secLines);
    contexts.push_back(ctx);
  }
  if (mergeFlag && contexts.size() > 1) {
    // Merge the first (base) Context with each subsequent section.
    tadah::core::Context base = contexts.front();
    for (std::size_t i = 1; i < contexts.size(); ++i) {
      contexts[i] = merge(base, contexts[i]);
    }
  }
  return contexts;
}

std::vector<std::vector<std::string>> MLIPUserConfigParser::splitSections(const std::vector<std::string>& lines) const {
  std::vector<std::vector<std::string>> sections;
  std::vector<std::string> current;
  for (const auto &lineOrig : lines) {
    std::string line = lineOrig;
    // Trim whitespace.
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    if (line.empty() || line[0] == '#')
      continue;
    if (core::to_upper(line).substr(0, 4) == "TASK") {
      // Start a new section.
      if (!current.empty()) {
        sections.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(line);
    }
  }
  if (!current.empty()) {
    sections.push_back(current);
  }
  std::cout << "Parsed sections:" << std::endl;
  for (const auto &sec : sections) {
    for (const auto &line : sec) {
      std::cout << line << std::endl;
    }
  }
  return sections;
}

tadah::core::Context MLIPUserConfigParser::merge(const tadah::core::Context &base, const tadah::core::Context &task) const {
  // Create a copy of the base Context.
  tadah::core::Context merged = base;
  // For each key in the task context, update the merged context.
  const auto &bpKeysMap = core::BlueprintParser::getKeysMapRef();
  const auto &taskKeyValues = task.getMapRef();
  for (const auto &pair : taskKeyValues) {
    const std::string &key = pair.first;
    bool allowed = bpKeysMap.find(key) != bpKeysMap.end();
    if (!allowed) continue; // overwrite only allowed keys, not internal ones
    const std::vector<std::string> &values = pair.second;
    // Add or update the key in the merged context.
    if (merged.contains(key)) {
      // Here we simply replace it with the task's values.
      core::Logger::instance().log(
          core::LogLevel::WARNING,
          "MLIPUserConfigParser::merge -> Replacing key: ", key);
      core::Logger::instance().log(
          core::LogLevel::WARNING,
          "MLIPUserConfigParser::merge -> Old values: ", merged(key));
      core::Logger::instance().log(
          core::LogLevel::WARNING,
          "MLIPUserConfigParser::merge -> New values: ", values);
      merged.remove(key);
    }
    merged.add(key, values);
  }
  return merged;
}

} // namespace mlip
} // namespace tadah
