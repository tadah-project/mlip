#ifndef TADAH_MLIP_TRAINING_RECORD_H
#define TADAH_MLIP_TRAINING_RECORD_H

#include <tadah/core/context.h>

#include <ostream>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

class KeyProvenance;

/** @file training_record.h
 *  The training record of a potential file (MK's decision of 2026-10-14):
 *  the keys a potential does not use -- the training datasets and the
 *  inputs of the run that wrote it -- kept, for the user's record only, as
 *  a comment block right above the key lines:
 *
 *      # --- training record (not read by tadah) ---
 *      # DBFILE db1.tadah db2.tadah
 *      # VALIDATION config.val
 *      # --- end of training record ---
 *
 *  Every reader of a potential (core::CoreUserConfigParser: tadah predict,
 *  tadah train -c, tadah hpo -c, refit --edit, pair_style tadah) skips a
 *  line whose first non-blank character is '#', so the block is never
 *  read: a potential loads, predicts and retrains as one without it. Only
 *  refit --edit reads it, with read_training_record(), to copy it into the
 *  edit. */

/** The keys a potential file keeps only in its training record, in the
 *  order the record lists them: DBFILE (the training datasets), VALIDATION
 *  and HPOTARGET (the validation and target files of a tadah hpo run),
 *  UNCERTAINTY (the train task's switch for weights.tadah), VERBOSE,
 *  NUMERIC and OUTFILE (the run's verbosity and the writer's settings),
 *  AUDIT, AUDIT_SAMPLE and HEALTH_LOG (the run's diagnostics). The one
 *  list both rules read: no key line of these is written
 *  (remove_training_record_keys), and the record lists the run's values
 *  of these (training_record). */
const std::vector<std::string> &training_record_keys();

/** Removes every key of training_record_keys() from @p pot. */
void remove_training_record_keys(core::Context &pot);

/** The lines of a training record, without their leading "# ". */
struct TrainingRecord {
  /** One line per entry, in the order the block lists them: "KEY v1 v2
   *  ..." or, after refit --edit, "edited by tadah refit --edit from
   *  <file>". */
  std::vector<std::string> lines;
  /** Whether the record has no line (its block is not written). */
  bool empty() const { return lines.empty(); }
};

/** The training record of a run whose configuration is @p config: one
 *  line "KEY v1 v2 ..." for each key of training_record_keys() that
 *  @p config stores, in that order, with the stored tokens as they are
 *  (the values as the configuration wrote them, never re-rendered:
 *  NUMERIC governs only the key lines). With @p provenance (the task's
 *  KeyProvenance), a key whose value is the blueprint default the task
 *  factory added (KeyOrigin::Kind::Default: VERBOSE 1, the default OUTFILE
 *  of a task) is left out, as no one wrote it -- the rule of
 *  engines::refit_key_configured; without it every stored key is the
 *  configuration's. Take it when the run starts, before any code inserts
 *  a default into the configuration. */
TrainingRecord training_record(const core::Context &config,
                               const KeyProvenance *provenance = nullptr);

/** The opening line of the block:
 *  "# --- training record (not read by tadah) ---". */
extern const char *const kTrainingRecordOpen;
/** The end line of the block: "# --- end of training record ---". */
extern const char *const kTrainingRecordEnd;

/** Writes the block of @p record: kTrainingRecordOpen, one "# <line>" per
 *  line, kTrainingRecordEnd, each ended by '\n'. Nothing for an empty
 *  record. */
void write_training_record(std::ostream &os, const TrainingRecord &record);

/** The training record of the potential file @p path: the lines of its
 *  block (or blocks, in order), without their leading '#' and the one
 *  blank after it; an empty record when the file has none (a potential
 *  of an earlier version). Throws std::runtime_error naming the file and
 *  the line when the file cannot be read, when an opening line has no end
 *  line after it, and when a line inside a block is not a comment (every
 *  reader of the file would read it as a key). refit --edit copies its
 *  input's record with it. */
TrainingRecord read_training_record(const std::string &path);

/** Writes the potential @p pot: the training record @p record
 *  (write_training_record), then pot.write_potential(os, sig_digits) of a
 *  copy of @p pot without the keys of training_record_keys(). The body of
 *  every potential file tadah writes (TrainEngine::dumpPotFile: tadah
 *  train, tadah refit and refit --edit; Tadah.HPO's
 *  PotHeaderWriter::write_potential_body: best_pot.tadah, its rank-local
 *  mirrors, the BDUMP and DUMP archives). */
void write_potential_with_record(std::ostream &os, const core::Context &pot,
                                 const TrainingRecord &record,
                                 int sig_digits);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_TRAINING_RECORD_H
