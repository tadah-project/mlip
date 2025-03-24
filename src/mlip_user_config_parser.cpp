#include "tadah/mlip/mlip_user_config_parser.h"
#include "tadah/core/core_user_config_parser.h"
#include "tadah/core/logger.h"
#include "tadah/core/utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace tadah {
namespace mlip {

static std::string to_upper(const std::string &s) {
  std::string result = s;
  std::transform(result.begin(), result.end(), result.begin(), ::toupper);
  return result;
}

MLIPUserConfigParser::MLIPUserConfigParser(const tadah::core::IUserConfigParser* backend)
  : backendParser_(backend)
{
  // If no backend is provided, use the default CoreUserConfigParser.
  if (!backendParser_) {
    backendParser_ = new tadah::core::CoreUserConfigParser();
    // Note: In production code, consider using a unique_ptr/shared_ptr.
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
  // The first section is considered the base contexturation.
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
    if (to_upper(line).substr(0, 4) == "TASK") {
      // Start a new section.
      if (!current.empty()) {
        sections.push_back(current);
        current.clear();
      }
      // Optionally, store the TASK command line itself in the section.
      current.push_back(line);
    } else {
      current.push_back(line);
    }
  }
  if (!current.empty()) {
    sections.push_back(current);
  }
  return sections;
}

tadah::core::Context MLIPUserConfigParser::merge(const tadah::core::Context &base, const tadah::core::Context &override) const {
  // Create a copy of the base Context.
  tadah::core::Context merged = base;
  // For each key in the override context, update the merged context.
  // (Assume that Context provides methods to iterate through keys.)
  // For simplicity, we use serialize/deserialize to simulate merging.
  // In production, one would iterate through keys and update accordingly.
  std::istringstream iss(tadah::core::to_string(override.serialize()));
  std::string line;
  while (std::getline(iss, line)) {
    auto pos = line.find('=');
    if (pos == std::string::npos)
      continue;
    std::string key = line.substr(0, pos);
    std::string valStr = line.substr(pos+1);
    // In merging, keys in override replace those in base.
    merged.add(key, valStr);
  }
  return merged;
}

} // namespace mlip
} // namespace tadah

