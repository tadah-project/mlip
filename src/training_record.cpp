#include <tadah/mlip/training_record.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/key_provenance.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

const char *const kTrainingRecordOpen =
    "# --- training record (no key is read from it) ---";
const char *const kTrainingRecordEnd = "# --- end of training record ---";

const std::vector<std::string> &dataset_preparation_keys() {
  static const std::vector<std::string> keys = {
      "EFILTER",      "FFILTER",        "ESHIFT_ATOM", "ESHIFT_DBATOM",
      "EWEIGHT_TEMP", "ZERO_COM_FORCE", "WDBFILE",     "WDBFILE_AUTO"};
  return keys;
}

const std::vector<std::string> &keys_not_read_from_a_potential() {
  static const std::vector<std::string> keys = [] {
    std::vector<std::string> k = dataset_preparation_keys();
    k.push_back("UNCERTAINTY");
    return k;
  }();
  return keys;
}

const std::vector<std::string> &training_record_keys() {
  // The keys a potential does not use. Moving a key into the record (out
  // of the key lines of every potential tadah writes) is adding it here.
  static const std::vector<std::string> keys = [] {
    std::vector<std::string> k = {"DBFILE"};
    const std::vector<std::string> &data = dataset_preparation_keys();
    k.insert(k.end(), data.begin(), data.end());
    for (const char *key : {"VALIDATION", "HPOTARGET", "UNCERTAINTY",
                            "VERBOSE", "NUMERIC", "OUTFILE", "AUDIT",
                            "AUDIT_SAMPLE", "HEALTH_LOG"})
      k.push_back(key);
    return k;
  }();
  return keys;
}

namespace {

// Whether the tokens @p a and @p b say the same: equal numbers, when both
// are numbers, else equal words whatever their letter case.
bool same_token(const std::string &a, const std::string &b) {
  const auto number = [](const std::string &t, double &v) {
    if (t.empty()) return false;
    char *end = nullptr;
    v = std::strtod(t.c_str(), &end);
    return end != t.c_str() && *end == '\0';
  };
  double x = 0.0, y = 0.0;
  if (number(a, x) && number(b, y)) return x == y;
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  return true;
}

// Whether the stored @p tokens of @p key are its blueprint default: then
// the key means the same read or ignored.
bool at_blueprint_default(const std::string &key,
                          const std::vector<std::string> &tokens) {
  const auto &keys = core::BlueprintParser::getKeysMapRef();
  const auto found = keys.find(key);
  if (found == keys.end()) return false;
  const std::vector<std::string> &d = found->second.default_value;
  if (d.empty() || d.size() != tokens.size()) return false;
  for (std::size_t i = 0; i < d.size(); ++i)
    if (!same_token(tokens[i], d[i])) return false;
  return true;
}

// "KEY v1 v2".
std::string key_values(const std::string &key,
                       const std::vector<std::string> &tokens) {
  std::string s = key;
  for (const std::string &t : tokens) s += " " + t;
  return s;
}

// "a", "a and b", "a, b and c" (@p last "and"), or with "or".
std::string listed(const std::vector<std::string> &items,
                   const std::string &last) {
  std::string s;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i > 0) s += (i + 1 == items.size()) ? " " + last + " " : ", ";
    s += items[i];
  }
  return s;
}

// The long command-line option of @p key ("--efilter"), "" without one.
std::string option_of(const std::string &key) {
  const auto &keys = core::BlueprintParser::getKeysMapRef();
  const auto found = keys.find(key);
  return found == keys.end() ? std::string() : found->second.cli_long_flag;
}

const char *const kRecordNote =
    "(tadah keeps such keys in a potential's training record, not as key "
    "lines)";

}  // namespace

std::string ignore_potential_dataset_keys(core::Context &pot,
                                          const std::string &file) {
  std::vector<std::string> named;
  for (const std::string &key : dataset_preparation_keys()) {
    if (!pot.has_stored(key)) continue;
    const std::vector<std::string> tokens = pot(key);
    if (!at_blueprint_default(key, tokens))
      named.push_back(key_values(key, tokens));
    pot.remove(key);
  }
  if (named.empty()) return "";
  return "tadah predict: " + file + ": " + listed(named, "and") +
         " prepared the training data of the potential and " +
         (named.size() == 1 ? "is" : "are") +
         " not applied to the data predicted " + kRecordNote +
         "; --efilter, --ffilter and --zero-com-force filter the data "
         "predicted.";
}

bool potential_line_not_read(const std::set<std::string> &file_keys,
                             const std::string &key) {
  if (!file_keys.count("WEIGHTS")) return false;
  const std::vector<std::string> &keys = keys_not_read_from_a_potential();
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

namespace {

/* One key of a potential that was not read: its values as written. */
struct IgnoredKey {
  std::string key;
  std::vector<std::string> values;
};

/* The keys of the blueprint task @p task ("train", "hpo"), upper case. */
std::set<std::string> task_keys(const std::string &task) {
  const auto &tasks = core::BlueprintParser::getTasksMapRef();
  const auto found = tasks.find(core::to_upper(task));
  if (found == tasks.end())
    throw std::invalid_argument(
        "ignore_configuration_potential_lines: no blueprint task '" + task +
        "'");
  std::set<std::string> keys;
  for (const auto &k : found->second.blueprintKeys) {
    const std::size_t dot = k.keyName.rfind('.');
    keys.insert(core::to_upper(
        dot == std::string::npos ? k.keyName : k.keyName.substr(dot + 1)));
  }
  return keys;
}

/* The sentences of the notice for the keys @p ignored (in the order of
 * keys_not_read_from_a_potential(), none at its default) of the potential
 * @p subject ("the configuration file pot.train"); @p advice(keys,
 * options) says where the run's own go (keys: the named keys; options:
 * their command-line options), "" for none; @p eshift: whether the run's
 * configuration has ESHIFT, which overrules a toggle the run gives. */
template <class Advice>
std::string ignored_keys_notice(const std::string &who,
                                const std::string &subject,
                                const std::vector<IgnoredKey> &ignored,
                                Advice advice, bool eshift) {
  std::vector<std::string> prepared, keys, options;
  std::string uncertainty;  // "UNCERTAINTY true", "" when not named
  bool toggle_named = false;
  for (const IgnoredKey &k : ignored) {
    keys.push_back(k.key);
    if (!option_of(k.key).empty()) options.push_back(option_of(k.key));
    if (k.key == "UNCERTAINTY") {
      uncertainty = key_values(k.key, k.values);
      continue;
    }
    prepared.push_back(key_values(k.key, k.values));
    if (k.key == "ESHIFT_ATOM" || k.key == "ESHIFT_DBATOM") toggle_named = true;
  }
  std::string s = who + ": " + subject + " is a potential (it has WEIGHTS): ";
  const std::string asked =
      " asked the run that wrote it for weights.tadah, not this run, and is "
      "ignored";
  if (!prepared.empty()) {
    s += "its " + listed(prepared, "and") +
         " prepared the training data of the fit that wrote it, not of this "
         "run, and " +
         (prepared.size() == 1 ? "is" : "are") + " ignored " + kRecordNote +
         ".";
    if (!uncertainty.empty()) s += " Its " + uncertainty + asked + " too.";
  } else {
    s += "its " + uncertainty + asked + " " + kRecordNote + ".";
  }
  const std::string where = advice(keys, options);
  if (!where.empty()) s += " " + where;
  if (toggle_named && eshift) {
    if (!where.empty()) s += ".";
    s += " The ESHIFT of the configuration, the energy reference of the "
         "model, overrules ESHIFT_ATOM and ESHIFT_DBATOM: to derive ESHIFT "
         "from the data of this run, delete the ESHIFT line and give the run "
         "one of them";
  }
  // The advice and the toggle sentence end without their period.
  return s + " (the Training page of the documentation, Retraining from a "
             "potential).";
}

/* "it" or "one": what a run applies of the @p n keys an advice names. */
std::string one_of(std::size_t n) { return n == 1 ? "it" : "one"; }

/* "an" or "a" before the key @p key: a key that begins with UN is read as
 * the word it is ("an UNCERTAINTY"), any other letter by letter ("an
 * EFILTER", "an FFILTER", "a WDBFILE"). */
std::string article(const std::string &key) {
  if (key.rfind("UN", 0) == 0) return "an";
  const std::string vowel_sounds = "AEFHILMNORSX";
  return !key.empty() && vowel_sounds.find(key[0]) != std::string::npos
             ? "an"
             : "a";
}

/* Whether the run gives @p key itself: on the command line, or on a line
 * of the task file or of a configuration file without WEIGHTS (the parse
 * reads no line of a potential's keys of keys_not_read_from_a_potential(),
 * so such an origin is the run's own). */
bool run_gives(const KeyProvenance &provenance, const std::string &key) {
  const KeyOrigin::Kind kind = provenance.origin(key).kind;
  return kind == KeyOrigin::Kind::CommandLine ||
         kind == KeyOrigin::Kind::ConfigFile ||
         kind == KeyOrigin::Kind::TaskFile;
}

/* The ESHIFT toggle that is not @p key, "" when @p key is not a toggle. */
std::string other_toggle(const std::string &key) {
  if (key == "ESHIFT_ATOM") return "ESHIFT_DBATOM";
  if (key == "ESHIFT_DBATOM") return "ESHIFT_ATOM";
  return "";
}

/* The notice of the lines TaskFactory's parse did not read
 * (KeyProvenance::skipped_key_lines), per potential file: those of the
 * keys @p taken (the task's) only, the run's configuration @p config (the
 * parse's result: the run's own values); @p eshift as for
 * ignored_keys_notice. A line is left out when the run gives its key
 * itself (the run's own value applies), and a toggle's line also when the
 * run sets the other toggle true: the two choose one thing, the source of
 * the shift, and exclude each other, so the run's own toggle is its
 * choice (and adding the potential's would set both: "Multiple ESHIFT
 * sources set"). */
std::string skipped_lines_notice(const KeyProvenance &provenance,
                                 const core::Context &config,
                                 const std::string &who,
                                 const std::set<std::string> &taken,
                                 bool eshift) {
  const auto run_sets_true = [&](const std::string &key) {
    return run_gives(provenance, key) && config.has_stored(key) &&
           config.get<bool>(key);
  };
  // The potential files, in the order their lines were read, and their
  // keys the run does not give itself, not at their default.
  std::vector<std::string> sources;
  std::vector<std::vector<IgnoredKey>> per_source;
  for (const core::SkippedKeyLine &line : provenance.skipped_key_lines()) {
    if (run_gives(provenance, line.key)) continue;  // the run's own applies
    const std::string other = other_toggle(line.key);
    if (!other.empty() && run_sets_true(other))
      continue;  // the run's own toggle chose the source of the shift
    auto at = std::find(sources.begin(), sources.end(), line.source);
    if (at == sources.end()) {
      sources.push_back(line.source);
      per_source.emplace_back();
      at = sources.end() - 1;
    }
    std::vector<IgnoredKey> &keys = per_source[at - sources.begin()];
    auto k = std::find_if(keys.begin(), keys.end(),
                          [&](const IgnoredKey &i) { return i.key == line.key; });
    if (k == keys.end()) {
      keys.push_back({line.key, line.values});
    } else {
      k->values.insert(k->values.end(), line.values.begin(),
                       line.values.end());
    }
  }
  const std::vector<std::string> &order = keys_not_read_from_a_potential();
  const std::string &top = provenance.file_path();
  const bool task_file = provenance.file_kind() == KeyOrigin::Kind::TaskFile;
  std::string out;
  for (std::size_t f = 0; f < sources.size(); ++f) {
    std::vector<IgnoredKey> ignored;
    for (const std::string &key : order)
      for (const IgnoredKey &k : per_source[f])
        if (k.key == key && taken.count(k.key) &&
            !at_blueprint_default(k.key, k.values))
          ignored.push_back(k);
    if (ignored.empty()) continue;
    const std::string &file = sources[f];
    const bool included = file != top;
    const std::string kind = task_file ? "task file " : "configuration file ";
    std::string subject;
    if (!included)
      subject = task_file ? "a section of the task file " + top
                          : "the configuration file " + top;
    else
      subject = file + ", which the " + kind + top + " includes with CONFIG,";
    const auto advice = [&](const std::vector<std::string> &keys,
                            const std::vector<std::string> &options) {
      const std::string lines = listed(keys, "or");
      if (task_file)
        return "Write " + lines + " in " +
               (included ? "the task file " + top
                         : "a section of the task file " + top +
                               " without WEIGHTS") +
               " to apply " + one_of(keys.size()) + " to this run";
      std::string a = "Give ";
      if (!options.empty()) a += listed(options, "or") + " on the command line, or ";
      a += article(keys[0]) + " ";
      a += lines + " line in " +
           (included ? top
                     : "a configuration file that includes the potential "
                       "with CONFIG");
      if (!options.empty()) a += ",";
      return a + " to apply " + one_of(keys.size()) + " to this run";
    };
    const std::string notice =
        ignored_keys_notice(who, subject, ignored, advice, eshift);
    out += (out.empty() ? "" : " ") + notice;
  }
  return out;
}

}  // namespace

std::string ignore_configuration_potential_lines(
    core::Context &config, const KeyProvenance *provenance,
    const std::string &task) {
  const std::string who = "tadah " + task;
  const std::set<std::string> taken = task_keys(task);
  const bool eshift = config.has_stored("ESHIFT") && config.size("ESHIFT") > 0;
  if (provenance)
    return skipped_lines_notice(*provenance, config, who, taken, eshift);
  // A context a program built: one file.
  std::set<std::string> keys;
  for (const auto &kv : config.storage_snapshot())
    if (!kv.second.empty()) keys.insert(kv.first);
  std::vector<IgnoredKey> ignored;
  for (const std::string &key : keys_not_read_from_a_potential()) {
    if (!config.has_stored(key) || !potential_line_not_read(keys, key))
      continue;
    const std::vector<std::string> tokens = config(key);
    if (taken.count(key) && !at_blueprint_default(key, tokens))
      ignored.push_back({key, tokens});
    config.remove(key);
  }
  if (ignored.empty()) return "";
  return ignored_keys_notice(
      who, "the configuration", ignored,
      [](const std::vector<std::string> &names,
         const std::vector<std::string> &options) {
        if (options.empty()) return std::string();
        return "Give " + listed(options, "or") +
               " on the command line to apply " + one_of(names.size()) +
               " to this run";
      },
      eshift);
}

void remove_training_record_keys(core::Context &pot) {
  for (const std::string &key : training_record_keys())
    if (pot.has_stored(key)) pot.remove(key);
}

TrainingRecord training_record(const core::Context &config,
                               const KeyProvenance *provenance) {
  TrainingRecord record;
  // The stored tokens (has_stored and the snapshot never insert a
  // blueprint default).
  const auto stored = config.storage_snapshot();
  for (const std::string &key : training_record_keys()) {
    const auto it = stored.find(key);
    if (it == stored.end() || it->second.empty()) continue;
    if (provenance &&
        provenance->origin(key).kind == KeyOrigin::Kind::Default)
      continue;
    std::string line = key;
    for (const std::string &value : it->second) line += " " + value;
    record.lines.push_back(line);
  }
  return record;
}

void write_training_record(std::ostream &os, const TrainingRecord &record) {
  if (record.empty()) return;
  os << kTrainingRecordOpen << "\n";
  for (const std::string &line : record.lines) os << "# " << line << "\n";
  os << kTrainingRecordEnd << "\n";
}

namespace {

// @p line without the blanks at its end.
std::string rtrim(const std::string &line) {
  const std::size_t end = line.find_last_not_of(" \t\r");
  return end == std::string::npos ? std::string() : line.substr(0, end + 1);
}

}  // namespace

TrainingRecord read_training_record(const std::string &path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("Cannot read the training record of " + path +
                             ": the file cannot be opened.");
  std::vector<std::string> lines;
  for (std::string raw; std::getline(in, raw);) lines.push_back(rtrim(raw));
  const std::string open_quoted =
      "('" + std::string(kTrainingRecordOpen) + "')";
  const std::string end_quoted = "('" + std::string(kTrainingRecordEnd) + "')";
  // The refusal of the block that line @p opened_at opens and no end line
  // closes.
  const auto unclosed = [&](std::size_t opened_at) {
    return path + ": line " + std::to_string(opened_at) + " opens a "
           "training record " + open_quoted + " that no end line " +
           end_quoted + " closes";
  };
  TrainingRecord record;
  std::size_t opened_at = 0;  // the line opening the block; 0 outside one
  for (std::size_t i = 0; i < lines.size(); ++i) {
    const std::string &line = lines[i];
    const std::size_t n = i + 1;
    if (opened_at == 0) {
      if (line == kTrainingRecordOpen)
        opened_at = n;
      else if (line == kTrainingRecordEnd)
        throw std::runtime_error(
            path + ": line " + std::to_string(n) + " ends a training "
            "record " + end_quoted + " that no opening line " + open_quoted +
            " opens: restore the opening line before the record, or delete "
            "the end line.");
      continue;
    }
    if (line == kTrainingRecordEnd) {
      opened_at = 0;
      continue;
    }
    if (line == kTrainingRecordOpen)
      throw std::runtime_error(
          unclosed(opened_at) + " before line " + std::to_string(n) +
          ", which opens another: restore the end line after the record, "
          "before line " + std::to_string(n) + ", or delete the record.");
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos) continue;  // a blank line
    if (line[first] != '#') {
      // A line that is not a comment is inside the block only when an end
      // line follows it before any opening line. Every writer puts the key
      // lines right after the record, so without one the block has lost
      // its end line and this is the potential's first key line.
      std::size_t closed_at = 0;
      for (std::size_t j = i + 1; j < lines.size(); ++j) {
        if (lines[j] == kTrainingRecordOpen) break;
        if (lines[j] == kTrainingRecordEnd) {
          closed_at = j + 1;
          break;
        }
      }
      if (closed_at == 0)
        throw std::runtime_error(
            unclosed(opened_at) + " before line " + std::to_string(n) +
            ", the first line after it that is not a comment: restore the "
            "end line after the record, before line " + std::to_string(n) +
            ", or delete the record.");
      throw std::runtime_error(
          path + ": line " + std::to_string(n) + ", inside the training "
          "record that line " + std::to_string(opened_at) + " opens and "
          "line " + std::to_string(closed_at) + " ends, is not a comment: "
          "every reader of the potential reads it as a key. Begin it with "
          "'#' if it belongs to the record, or move it after the end line.");
    }
    std::string text = line.substr(first + 1);
    if (!text.empty() && text[0] == ' ') text.erase(0, 1);
    if (!text.empty()) record.lines.push_back(text);
  }
  if (opened_at != 0)
    throw std::runtime_error(unclosed(opened_at) +
                             ": restore the end line after the record, or "
                             "delete the record.");
  return record;
}

void write_potential_with_record(std::ostream &os, const core::Context &pot,
                                 const TrainingRecord &record,
                                 int sig_digits) {
  write_training_record(os, record);
  core::Context written = pot;
  remove_training_record_keys(written);
  written.write_potential(os, sig_digits);
}

}  // namespace mlip
}  // namespace tadah
