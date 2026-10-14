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
