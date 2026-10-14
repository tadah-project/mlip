#include <tadah/mlip/key_provenance.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/utils.h>

#include <algorithm>
#include <utility>

namespace tadah {
namespace mlip {

namespace {
/* "-F/--force": the option names the blueprint gives the key. */
std::string option_names(const std::string &uKey) {
  const auto &keys = core::BlueprintParser::getKeysMapRef();
  const auto it = keys.find(uKey);
  if (it == keys.end()) return "";
  const std::string &s = it->second.cli_short_flag;
  const std::string &l = it->second.cli_long_flag;
  if (!s.empty() && !l.empty()) return s + "/" + l;
  return s.empty() ? l : s;
}

std::string join(const std::vector<std::string> &values) {
  std::string out;
  for (std::size_t i = 0; i < values.size(); ++i)
    out += (i ? " " : "") + values[i];
  return out;
}
} // namespace

std::string KeyOrigin::describe(const std::string &key) const {
  switch (kind) {
  case Kind::CommandLine: {
    std::string out = "command line";
    if (!flag.empty()) out += " " + flag;
    if (!overridden.empty())
      out += "; overrides " + core::to_upper(key) + " " + join(overridden) +
             " in " + file;
    return out;
  }
  case Kind::ConfigFile:
    return "configuration file " + file;
  case Kind::TaskFile:
    return "task file " + file;
  case Kind::Default:
    return "default";
  case Kind::NotRecorded:
    break;
  }
  return "source not recorded";
}

KeyProvenance::KeyProvenance() = default;
KeyProvenance::KeyProvenance(const KeyProvenance &) = default;
KeyProvenance::KeyProvenance(KeyProvenance &&) = default;
KeyProvenance &KeyProvenance::operator=(const KeyProvenance &) = default;
KeyProvenance &KeyProvenance::operator=(KeyProvenance &&) = default;
KeyProvenance::~KeyProvenance() = default;

KeyProvenance::KeyProvenance(core::Context command_line, core::Context file,
                             std::string file_path,
                             KeyOrigin::Kind file_kind,
                             std::vector<std::string> defaults,
                             std::vector<core::SkippedKeyLine> skipped)
    : command_line_(std::move(command_line)), file_(std::move(file)),
      file_path_(std::move(file_path)), file_kind_(file_kind),
      skipped_(std::move(skipped)), defaults_(std::move(defaults)) {
  for (auto &k : defaults_) k = core::to_upper(k);
  std::sort(defaults_.begin(), defaults_.end());
  defaults_.erase(std::unique(defaults_.begin(), defaults_.end()),
                  defaults_.end());
}

KeyOrigin KeyProvenance::origin(const std::string &key) const {
  const std::string uKey = core::to_upper(key);
  KeyOrigin o;
  // has_stored(), not contains(): contains() inserts the blueprint default.
  if (command_line_.has_stored(uKey)) {
    o.kind = KeyOrigin::Kind::CommandLine;
    o.flag = option_names(uKey);
    if (file_.has_stored(uKey)) {
      o.file = file_path_;
      o.overridden = file_(uKey);
    }
    return o;
  }
  if (file_.has_stored(uKey)) {
    o.kind = file_kind_;
    o.file = file_path_;
    return o;
  }
  if (std::binary_search(defaults_.begin(), defaults_.end(), uKey))
    o.kind = KeyOrigin::Kind::Default;
  return o;
}

} // namespace mlip
} // namespace tadah
