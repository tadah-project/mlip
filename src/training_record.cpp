#include <tadah/mlip/training_record.h>

#include <tadah/mlip/key_provenance.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

const char *const kTrainingRecordOpen =
    "# --- training record (not read by tadah) ---";
const char *const kTrainingRecordEnd = "# --- end of training record ---";

const std::vector<std::string> &training_record_keys() {
  // The keys a potential does not use. Moving a key into the record (out
  // of the key lines of every potential tadah writes) is adding it here.
  static const std::vector<std::string> keys = {
      "DBFILE",  "VALIDATION", "HPOTARGET", "UNCERTAINTY", "VERBOSE",
      "NUMERIC", "OUTFILE",    "AUDIT",     "AUDIT_SAMPLE", "HEALTH_LOG"};
  return keys;
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
  TrainingRecord record;
  std::size_t n = 0, opened_at = 0;
  bool in_block = false;
  for (std::string raw; std::getline(in, raw);) {
    ++n;
    const std::string line = rtrim(raw);
    if (!in_block) {
      if (line == kTrainingRecordOpen) {
        in_block = true;
        opened_at = n;
      }
      continue;
    }
    if (line == kTrainingRecordEnd) {
      in_block = false;
      continue;
    }
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos) continue;  // a blank line
    if (line[first] != '#')
      throw std::runtime_error(
          path + ": line " + std::to_string(n) + ", inside the training "
          "record that line " + std::to_string(opened_at) + " opens, is not "
          "a comment: every reader of the potential reads it as a key. "
          "Begin each line of the record with '#', or end the record ('" +
          std::string(kTrainingRecordEnd) + "') before it.");
    std::string text = line.substr(first + 1);
    if (!text.empty() && text[0] == ' ') text.erase(0, 1);
    if (!text.empty()) record.lines.push_back(text);
  }
  if (in_block)
    throw std::runtime_error(
        path + ": line " + std::to_string(opened_at) + " opens a training "
        "record ('" + std::string(kTrainingRecordOpen) + "') that no end "
        "line ('" + std::string(kTrainingRecordEnd) + "') closes: restore "
        "the end line after the record, or delete the record.");
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
