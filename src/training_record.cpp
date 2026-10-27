#include <tadah/mlip/training_record.h>

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/core_user_config_parser.h>  // core::run_setting_keys
#include <tadah/core/utils.h>
#include <tadah/mlip/key_provenance.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
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
      "EFILTER",        "FFILTER", "ESHIFT_ATOM", "ESHIFT_DBATOM",
      "ZERO_COM_FORCE", "WDBFILE"};
  return keys;
}

const std::vector<std::string> &keys_not_read_from_a_potential() {
  // The record's keys but DBFILE (a retrain adds its own DBFILE line to a
  // copy of the potential): one list.
  static const std::vector<std::string> keys = [] {
    const std::vector<std::string> &record = training_record_keys();
    std::vector<std::string> k;
    for (const std::string &key : record)
      if (key != "DBFILE") k.push_back(key);
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
    for (const char *key : {"VALIDATION", "HPOTARGET", "CHECK", "VERBOSE",
                            "NUMERIC", "OUTFILE", "AUDIT", "AUDIT_SAMPLE",
                            "HEALTH_LOG"})
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

// Whether @p tokens are the default @p d (not empty).
bool same_tokens(const std::vector<std::string> &d,
                 const std::vector<std::string> &tokens) {
  if (d.empty() || d.size() != tokens.size()) return false;
  for (std::size_t i = 0; i < d.size(); ++i)
    if (!same_token(tokens[i], d[i])) return false;
  return true;
}

// Whether the stored @p tokens of @p key are its blueprint default: then
// the key means the same read or ignored.
bool at_blueprint_default(const std::string &key,
                          const std::vector<std::string> &tokens) {
  const auto &keys = core::BlueprintParser::getKeysMapRef();
  const auto found = keys.find(key);
  if (found == keys.end()) return false;
  return same_tokens(found->second.default_value, tokens);
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

/* The key name @p key_name of a blueprint key ("TASKS.train.keys.outfile",
 * "outfile"), upper case. */
std::string key_of(const std::string &key_name) {
  const std::size_t dot = key_name.rfind('.');
  return core::to_upper(dot == std::string::npos ? key_name
                                                 : key_name.substr(dot + 1));
}

/* What a run of the blueprint task @p task reads of the keys of
 * keys_not_read_from_a_potential(): the keys of TASKS.<task> and of its
 * input groups (the task's command-line options: tadah hpo's VALIDATION
 * and HPOTARGET are group keys), with the task's own default where it
 * defines one (OUTFILE pot.tadah of tadah train), and the run settings
 * that tadah train and tadah hpo read from any configuration although no
 * option of theirs sets them: NUMERIC (the significant digits of the
 * potential files they write), AUDIT and AUDIT_SAMPLE (the pre-flight
 * audit) and, for tadah hpo, HEALTH_LOG (the health monitor of its
 * evaluations). */
struct TaskKeys {
  std::set<std::string> read;     // every key the run reads
  std::set<std::string> options;  // the keys the task has an option for
  std::map<std::string, std::vector<std::string>> defaults;  // the task's
};

/* The blueprint name of the task @p task as a user writes it: its command
 * upper case, then its subcommand ("train" -> "TRAIN", "analysis
 * check_bounds" -> "ANALYSIS.check_bounds"). */
std::string blueprint_task_name(const std::string &task) {
  std::istringstream words(task);
  std::string name, word;
  while (words >> word)
    name += name.empty() ? core::to_upper(word) : "." + word;
  return name;
}

TaskKeys task_keys(const std::string &task) {
  const auto &tasks = core::BlueprintParser::getTasksMapRef();
  const std::string name = blueprint_task_name(task);
  const auto found = tasks.find(name);
  if (found == tasks.end())
    throw std::invalid_argument(
        "ignore_configuration_potential_lines: no blueprint task '" + task +
        "'");
  TaskKeys t;
  const auto add = [&t](const core::BlueprintKey &k) {
    const std::string key = key_of(k.keyName);
    t.read.insert(key);
    t.options.insert(key);
    if (!k.default_value.empty()) t.defaults[key] = k.default_value;
  };
  for (const auto &k : found->second.blueprintKeys) add(k);
  for (const auto &g : found->second.groups)
    for (const auto &k : g.blueprintKeys) add(k);
  const std::string &up = name;
  if (up == "TRAIN" || up == "HPO")
    for (const char *key : {"NUMERIC", "AUDIT", "AUDIT_SAMPLE"})
      t.read.insert(key);
  if (up == "HPO") t.read.insert("HEALTH_LOG");
  return t;
}

/* Whether @p tokens of @p key are its default for the task @p t: the
 * task's own default when it defines one, else the key's. */
bool at_task_default(const TaskKeys &t, const std::string &key,
                     const std::vector<std::string> &tokens) {
  const auto d = t.defaults.find(key);
  if (d != t.defaults.end()) return same_tokens(d->second, tokens);
  return at_blueprint_default(key, tokens);
}

/* Where the ESHIFT of the run's configuration came from, for the sentence
 * on the toggles: none, a line of a file (or a context without a
 * provenance), the option --eshift, or the option over a line of the
 * configuration. */
enum class EshiftSource { None, Line, Option, OptionOverLine };

/* The sentence that says when ESHIFT is derived from the data of the run
 * when a potential's toggle is named and the run's configuration has
 * ESHIFT (@p source), by where that ESHIFT came from; "" for none. Begins
 * with a blank, ends without its period. */
std::string eshift_sentence(EshiftSource source) {
  const std::string overrules =
      ", the energy reference of the model, overrules ESHIFT_ATOM and "
      "ESHIFT_DBATOM: ESHIFT is derived from the data of this run when ";
  switch (source) {
  case EshiftSource::None:
    return "";
  case EshiftSource::Line:
    return " The ESHIFT of the configuration" + overrules +
           "the ESHIFT line is deleted and the run is given one of them";
  case EshiftSource::Option:
    return " The ESHIFT given with --eshift" + overrules +
           "--eshift is left out and the run is given one of them";
  case EshiftSource::OptionOverLine:
    return " The ESHIFT given with --eshift" + overrules +
           "--eshift is left out, the ESHIFT line is deleted and the run is "
           "given one of them";
  }
  return "";
}

/* The sentences of the notice for the keys @p ignored (in the order of
 * keys_not_read_from_a_potential(), none at its default) of the potential
 * @p subject ("the configuration file pot.train"); @p advice(keys,
 * optioned, options) says where the run's own go (keys: the named keys;
 * optioned: those of them the task @p t has an option of; options: those
 * options, in the same order), "" for none; @p eshift: where the
 * ESHIFT of the run's configuration came from, which overrules a toggle
 * the run gives. Two kinds of key, each a sentence: the keys that prepared
 * the training data, and the run settings; the first ends with the note on
 * the training record, the other with "too". */
template <class Advice>
std::string ignored_keys_notice(const std::string &who,
                                const std::string &subject,
                                const std::vector<IgnoredKey> &ignored,
                                const TaskKeys &t, Advice advice,
                                EshiftSource eshift) {
  const std::vector<std::string> &data = dataset_preparation_keys();
  std::vector<std::string> prepared, settings, keys, optioned, options;
  bool toggle_named = false;
  for (const IgnoredKey &k : ignored) {
    keys.push_back(k.key);
    if (t.options.count(k.key) && !option_of(k.key).empty()) {
      optioned.push_back(k.key);
      options.push_back(option_of(k.key));
    }
    if (std::find(data.begin(), data.end(), k.key) != data.end()) {
      prepared.push_back(key_values(k.key, k.values));
      if (k.key == "ESHIFT_ATOM" || k.key == "ESHIFT_DBATOM")
        toggle_named = true;
    } else {
      settings.push_back(key_values(k.key, k.values));
    }
  }
  // Each sentence without its "its" and its end.
  std::vector<std::string> sentences;
  if (!prepared.empty())
    sentences.push_back(listed(prepared, "and") +
                        " prepared the training data of the fit that wrote "
                        "it, not of this run, and " +
                        (prepared.size() == 1 ? "is" : "are") + " ignored");
  if (!settings.empty())
    sentences.push_back(listed(settings, "and") +
                        (settings.size() == 1 ? " was a setting" : " were settings") +
                        " of the run that wrote it, not of this run, and " +
                        (settings.size() == 1 ? "is" : "are") + " ignored");
  std::string s = who + ": " + subject + " is a potential (it has WEIGHTS): ";
  for (std::size_t i = 0; i < sentences.size(); ++i)
    s += i == 0 ? "its " + sentences[i] + " " + kRecordNote + "."
                : " Its " + sentences[i] + " too.";
  const std::string where = advice(keys, optioned, options);
  if (!where.empty()) s += " " + where;
  if (toggle_named && eshift != EshiftSource::None) {
    if (!where.empty()) s += ".";
    s += eshift_sentence(eshift);
  }
  // The advice and the toggle sentence end without their period.
  return s + " (the Training page of the documentation, Retraining from a "
             "potential).";
}

/* Whether @p c is a vowel letter. */
bool vowel(char c) { return std::string("AEIOU").find(c) != std::string::npos; }

/* "an" or "a" before the key @p key as it is read: one that begins with a
 * vowel takes "an" ("an EFILTER", "an AUDIT", "an OUTFILE"); one that begins with a consonant and a vowel is read as a
 * word and takes "a" ("a VERBOSE", "a NUMERIC", "a HEALTH_LOG", "a
 * ZERO_COM_FORCE"); one that begins with two consonants is read letter by
 * letter: "an" before a letter whose name begins with a vowel sound ("an
 * FFILTER", "an HPOTARGET"), else "a" ("a WDBFILE"). */
std::string article(const std::string &key) {
  if (key.empty()) return "a";
  if (vowel(key[0])) return "an";
  if (key.size() > 1 && vowel(key[1])) return "a";
  const std::string vowel_sounds = "FHLMNRSX";
  return vowel_sounds.find(key[0]) != std::string::npos ? "an" : "a";
}

/* Whether the run gives @p key itself: on the command line, or on a line
 * of a configuration file without WEIGHTS (the parse reads no line of a
 * potential's keys of keys_not_read_from_a_potential(), so such an origin
 * is the run's own). */
bool run_gives(const KeyProvenance &provenance, const std::string &key) {
  const KeyOrigin::Kind kind = provenance.origin(key).kind;
  return kind == KeyOrigin::Kind::CommandLine ||
         kind == KeyOrigin::Kind::ConfigFile;
}

/* The ESHIFT toggle that is not @p key, "" when @p key is not a toggle. */
std::string other_toggle(const std::string &key) {
  if (key == "ESHIFT_ATOM") return "ESHIFT_DBATOM";
  if (key == "ESHIFT_DBATOM") return "ESHIFT_ATOM";
  return "";
}

/* The notice of the lines TaskFactory's parse did not read
 * (KeyProvenance::skipped_key_lines), per potential file: those of the
 * keys the run reads (@p t), the run's configuration @p config (the
 * parse's result: the run's own values); @p eshift as for
 * ignored_keys_notice. A line is left out when the run gives its key
 * itself (the run's own value applies), and a toggle's line also when the
 * run sets the other toggle true: the two choose one thing, the source of
 * the shift, and exclude each other, so the run's own toggle is its
 * choice (and adding the potential's would set both: "Multiple ESHIFT
 * sources set"). */
std::string skipped_lines_notice(const KeyProvenance &provenance,
                                 const core::Context &config,
                                 const std::string &who, const TaskKeys &t,
                                 EshiftSource eshift) {
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
  std::string out;
  for (std::size_t f = 0; f < sources.size(); ++f) {
    std::vector<IgnoredKey> ignored;
    for (const std::string &key : order)
      for (const IgnoredKey &k : per_source[f])
        if (k.key == key && t.read.count(k.key) &&
            !at_task_default(t, k.key, k.values))
          ignored.push_back(k);
    if (ignored.empty()) continue;
    const std::string &file = sources[f];
    const bool included = file != top;
    const std::string subject =
        included ? file + ", which the configuration file " + top +
                       " includes with CONFIG,"
                 : "the configuration file " + top;
    // Where the run's own values come from. Every key has an option: "This
    // run takes its own EFILTER or WDBFILE from --efilter or --wdbfile on
    // the command line, or from an EFILTER or WDBFILE line in ...". Some
    // have: the line first, for every key, then the keys that have an
    // option with their options, "This run takes its own VERBOSE or
    // NUMERIC from a VERBOSE or NUMERIC line in ..., or VERBOSE from
    // --verbose on the command line" (an option is never offered for a key
    // it does not set).
    const auto advice = [&](const std::vector<std::string> &keys,
                            const std::vector<std::string> &optioned,
                            const std::vector<std::string> &options) {
      const std::string named = listed(keys, "or");
      const std::string line =
          article(keys[0]) + " " + named + " line in " +
          (included ? top
                    : "a configuration file that includes the potential "
                      "with CONFIG");
      const std::string head = "This run takes its own " + named + " from ";
      if (options.empty()) return head + line;
      const std::string command_line =
          listed(options, "or") + " on the command line";
      if (optioned.size() == keys.size())
        return head + command_line + ", or from " + line;
      return head + line + ", or " + listed(optioned, "or") + " from " +
             command_line;
    };
    const std::string notice =
        ignored_keys_notice(who, subject, ignored, t, advice, eshift);
    out += (out.empty() ? "" : " ") + notice;
  }
  return out;
}

}  // namespace

std::string ignore_configuration_potential_lines(
    core::Context &config, const KeyProvenance *provenance,
    const std::string &task) {
  const std::string who = "tadah " + task;
  const TaskKeys t = task_keys(task);
  EshiftSource eshift = EshiftSource::None;
  if (config.has_stored("ESHIFT") && config.size("ESHIFT") > 0) {
    eshift = EshiftSource::Line;
    if (provenance) {
      const KeyOrigin o = provenance->origin("ESHIFT");
      if (o.kind == KeyOrigin::Kind::CommandLine)
        eshift = o.overridden.empty() ? EshiftSource::Option
                                      : EshiftSource::OptionOverLine;
    }
  }
  if (provenance)
    return skipped_lines_notice(*provenance, config, who, t, eshift);
  // A context a program built: one file. Its run settings are the
  // program's own -- no parse stores a potential's
  // (core::run_setting_keys()) -- and are kept; the other keys of the list
  // are taken as the potential's.
  const std::vector<std::string> &settings = core::run_setting_keys();
  std::set<std::string> keys;
  for (const auto &kv : config.storage_snapshot())
    if (!kv.second.empty()) keys.insert(kv.first);
  std::vector<IgnoredKey> ignored;
  for (const std::string &key : keys_not_read_from_a_potential()) {
    if (std::find(settings.begin(), settings.end(), key) != settings.end())
      continue;
    if (!config.has_stored(key) || !potential_line_not_read(keys, key))
      continue;
    const std::vector<std::string> tokens = config(key);
    if (t.read.count(key) && !at_task_default(t, key, tokens))
      ignored.push_back({key, tokens});
    config.remove(key);
  }
  if (ignored.empty()) return "";
  return ignored_keys_notice(
      who, "the configuration", ignored, t,
      // The keys that have an option, with their options (a key without
      // one has no command-line source to name).
      [](const std::vector<std::string> &,
         const std::vector<std::string> &optioned,
         const std::vector<std::string> &options) {
        if (options.empty()) return std::string();
        return "This run takes its own " + listed(optioned, "or") +
               " from " + listed(options, "or") + " on the command line";
      },
      eshift);
}

std::string missing_run_setting(const std::string &task,
                                const std::string &key,
                                const std::string &what,
                                const KeyProvenance *provenance) {
  // The potential files whose lines of the key the parse did not read, in
  // the order read, with those lines ("HPOTARGET config.hpo").
  std::vector<std::string> sources;
  std::vector<std::vector<std::string>> lines;
  if (provenance)
    for (const core::SkippedKeyLine &line : provenance->skipped_key_lines()) {
      if (line.key != key) continue;
      auto at = std::find(sources.begin(), sources.end(), line.source);
      if (at == sources.end()) {
        sources.push_back(line.source);
        lines.emplace_back();
        at = sources.end() - 1;
      }
      lines[at - sources.begin()].push_back(key_values(line.key, line.values));
    }
  const std::string a_line = article(key) + " " + key + " line";
  std::string s =
      "tadah " + task + ": no " + key + ": " + what + " is given with ";
  const TaskKeys t = task_keys(task);
  const std::string option = option_of(key);
  if (t.options.count(key) && !option.empty()) s += option + ", or with ";
  s += a_line + " in a configuration file without WEIGHTS.";
  for (std::size_t f = 0; f < sources.size(); ++f) {
    s += " " + sources[f] + " is a potential (it has WEIGHTS): its " +
         listed(lines[f], "and") +
         (lines[f].size() == 1
              ? " line was a setting of the run that wrote it and is not read."
              : " lines were settings of the run that wrote it and are not "
                "read.");
  }
  return s;
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

TrainingRecord record_of_key_lines(
    const core::Context &pot,
    const std::vector<core::SkippedKeyLine> &not_read) {
  TrainingRecord record;
  const auto stored = pot.storage_snapshot();
  for (const std::string &key : training_record_keys()) {
    std::vector<std::string> values;
    const auto it = stored.find(key);
    if (it != stored.end() && !it->second.empty()) {
      values = it->second;
    } else {
      for (const core::SkippedKeyLine &line : not_read)
        if (line.key == key)
          values.insert(values.end(), line.values.begin(), line.values.end());
    }
    if (values.empty()) continue;
    record.lines.push_back(key_values(key, values));
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
            " opens: a file with the opening line before the record, or "
            "without the end line, is accepted.");
      continue;
    }
    if (line == kTrainingRecordEnd) {
      opened_at = 0;
      continue;
    }
    if (line == kTrainingRecordOpen)
      throw std::runtime_error(
          unclosed(opened_at) + " before line " + std::to_string(n) +
          ", which opens another: a file with the end line after the "
          "record, before line " + std::to_string(n) + ", or without the "
          "record, is accepted.");
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
            ", the first line after it that is not a comment: a file with "
            "the end line after the record, before line " +
            std::to_string(n) + ", or without the record, is accepted.");
      throw std::runtime_error(
          path + ": line " + std::to_string(n) + ", inside the training "
          "record that line " + std::to_string(opened_at) + " opens and "
          "line " + std::to_string(closed_at) + " ends, is not a comment: "
          "every reader of the potential reads it as a key. A line of the "
          "record begins with '#', and a key line comes after the end line.");
    }
    std::string text = line.substr(first + 1);
    if (!text.empty() && text[0] == ' ') text.erase(0, 1);
    if (!text.empty()) record.lines.push_back(text);
  }
  if (opened_at != 0)
    throw std::runtime_error(unclosed(opened_at) +
                             ": a file with the end line after the record, "
                             "or without the record, is accepted.");
  return record;
}

void write_potential_with_record(std::ostream &os, const core::Context &pot,
                                 const TrainingRecord &record,
                                 int sig_digits) {
  write_training_record(os, record);
  core::Context written = pot;
  remove_training_record_keys(written);
  written.write_potential(os, sig_digits);
  // The retired lines the parse of the potential's input recorded (a
  // potential of an earlier version that refit --edit edits): as they were
  // written, after the key lines, so that a fit from the written file
  // refuses them as it refuses its input. No fit writes a potential from a
  // Context that records one: every fit refuses it first.
  for (const core::RetiredKeyLine &l : pot.retired_keys()) os << l.text << '\n';
}

}  // namespace mlip
}  // namespace tadah
