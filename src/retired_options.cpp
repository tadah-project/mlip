#include <tadah/mlip/retired_options.h>

#include <tadah/core/utils.h>

#include <algorithm>

namespace tadah {
namespace mlip {

namespace {

/* The text with its whitespace runs made single blanks, trimmed. */
std::string one_line(const std::string &text) {
  std::string out;
  bool blank = false;
  for (char ch : text) {
    if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') {
      blank = !out.empty();
      continue;
    }
    if (blank) out += ' ';
    blank = false;
    out += ch;
  }
  return out;
}

/* The last part of a blueprint key name ("DATA.balance.rescale" ->
 * "rescale"). */
std::string local_name(const std::string &key_name) {
  const auto dot = key_name.rfind('.');
  return dot == std::string::npos ? key_name : key_name.substr(dot + 1);
}

}  // namespace

const core::BlueprintKey *retired_task_option(const std::string &task,
                                              const std::string &name) {
  const auto &tasks = core::BlueprintParser::getTasksMapRef();
  const auto it = tasks.find(task);
  if (it == tasks.end()) return nullptr;
  const std::string wanted = core::to_lower(name);
  auto match = [&](const core::BlueprintKey &k) {
    return k.isRetired() && core::to_lower(local_name(k.keyName)) == wanted;
  };
  const core::BlueprintKey *option = nullptr;
  for (const core::BlueprintKey &k : it->second.blueprintKeys)
    if (!option && match(k)) option = &k;
  for (const auto &group : it->second.groups)
    for (const core::BlueprintKey &k : group.blueprintKeys)
      if (!option && match(k)) option = &k;
  if (!option) return nullptr;
  // A key that is retired itself (EWEIGHT, --wdbfile-auto, --uncertainty:
  // the task's option inherits the key's `retired`) is refused as such,
  // whatever the command (retired_keys.h); an option the task retires with
  // a reason of its own (--error of tadah predict, whose key ERROR is
  // retired too) is refused with the task's text, which says what the
  // option gave there.
  const auto &keys = core::BlueprintParser::getKeysMapRef();
  const auto own = keys.find(core::to_upper(name));
  if (own != keys.end() && own->second.isRetired() &&
      own->second.retired.reason == option->retired.reason &&
      own->second.retired.conversion == option->retired.conversion)
    return nullptr;
  return option;
}

std::string retired_task_option_text(const std::string &task,
                                     const core::BlueprintKey &option) {
  std::string command = "tadah " + core::to_lower(task);
  std::replace(command.begin(), command.end(), '.', ' ');
  std::string name = option.cli_long_flag.empty()
                         ? core::to_upper(local_name(option.keyName))
                         : option.cli_long_flag;
  if (!option.cli_long_flag.empty() && !option.cli_short_flag.empty())
    name += " (" + option.cli_short_flag + ")";
  return name + " of " + command + " is retired: " +
         one_line(option.retired.reason) + " Replaced by: " +
         one_line(option.retired.replacement) + ". " +
         one_line(option.retired.conversion);
}

}  // namespace mlip
}  // namespace tadah
