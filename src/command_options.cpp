#include <tadah/mlip/command_options.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/utils.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

/* The last part of a blueprint key name, upper case ("PREDICT.input.dbfile"
 * -> "DBFILE"). */
std::string option_key(const std::string &key_name) {
  const auto dot = key_name.rfind('.');
  return core::to_upper(dot == std::string::npos ? key_name
                                                 : key_name.substr(dot + 1));
}

/* "A", "A and B", "A, B and C". */
std::string listed(const std::vector<std::string> &items) {
  std::string out;
  for (std::size_t i = 0; i < items.size(); ++i)
    out += (i == 0 ? "" : i + 1 == items.size() ? " and " : ", ") + items[i];
  return out;
}

}  // namespace

void require_command_options(const core::Context &run,
                             const std::string &task) {
  const core::BlueprintTask &bp =
      core::BlueprintParser::getTasksMapRef().at(task);
  std::set<std::string> options;
  for (const core::BlueprintKey &k : bp.blueprintKeys)
    options.insert(option_key(k.keyName));
  for (const core::BlueprintKeyGroup &g : bp.groups)
    for (const core::BlueprintKey &k : g.blueprintKeys)
      options.insert(option_key(k.keyName));
  const auto &internal = core::BlueprintParser::getInternalKeysMapRef();

  std::set<std::string> keys;
  for (const auto &kv : run.storage_snapshot()) {
    if (kv.second.empty() || kv.first.empty() || kv.first[0] == '@') continue;
    if (options.count(kv.first) || internal.count(kv.first)) continue;
    keys.insert(kv.first);
  }
  for (const core::RetiredKeyLine &line : run.retired_keys())
    if (!options.count(line.key)) keys.insert(line.key);
  std::set<std::string> blocks;
  for (const core::ConfigBlock &b : run.blocks()) blocks.insert(b.kind);
  if (keys.empty() && blocks.empty()) return;

  std::vector<std::string> names(keys.begin(), keys.end());
  for (const std::string &kind : blocks)
    names.push_back("the " + kind + " block");
  std::string command = "tadah " + core::to_lower(task);
  std::replace(command.begin(), command.end(), '.', ' ');
  throw std::runtime_error(
      command + ": " + listed(names) +
      (names.size() == 1 ? " is not an option of " : " are not options of ") +
      command +
      ": the model is the potential's, as it was fitted (its keys are those "
      "of the potential file)");
}

}  // namespace mlip
}  // namespace tadah
