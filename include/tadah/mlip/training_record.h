#ifndef TADAH_MLIP_TRAINING_RECORD_H
#define TADAH_MLIP_TRAINING_RECORD_H

#include <tadah/core/context.h>

#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

class KeyProvenance;

/** @file training_record.h
 *  The training record of a potential file (MK's decisions of 2026-10-14):
 *  the keys a potential does not use -- the training datasets, the keys
 *  that prepared the training data and the inputs of the run that wrote it
 *  -- kept, for the user's record only, as a comment block right above the
 *  key lines:
 *
 *      # --- training record (no key is read from it) ---
 *      # DBFILE db1.tadah db2.tadah
 *      # WDBFILE 1 0.5
 *      # VALIDATION config.val
 *      # --- end of training record ---
 *
 *  A potential's key lines are what is needed to use the model: the model's
 *  keys and its fit, the resolved ESHIFT values and LSCALE.
 *
 *  Every loader of a potential (core::CoreUserConfigParser: tadah predict,
 *  tadah train -c, tadah hpo -c, refit --edit's read of its input,
 *  pair_style tadah) skips a line whose first non-blank character is '#',
 *  so no key is read from the block: a potential loads, predicts and
 *  retrains as one without it. refit --edit alone reads the block, with
 *  read_training_record(), to copy it into the edit, and refuses an input
 *  whose block is malformed. */

/** The keys that prepared the training data, in the order their
 *  transforms apply (apply_load_transforms): EFILTER and FFILTER (which
 *  structures were fitted), ESHIFT_ATOM and ESHIFT_DBATOM (how the run
 *  derived ESHIFT from its data: the values they produced are the
 *  potential's ESHIFT line), EWEIGHT_TEMP (the Boltzmann weights),
 *  ZERO_COM_FORCE (the reference forces), WDBFILE and WDBFILE_AUTO (the
 *  weight of each dataset). The model does not use them (MK's decision of
 *  2026-10-14, option 2): a potential lists them in its training record
 *  (training_record_keys), tadah predict ignores them in a potential
 *  (ignore_potential_dataset_keys) and a retrain from a potential does not
 *  read the potential's lines of them (potential_dataset_key_line). LSCALE
 *  and ESHIFT, the model's length scale and energy reference, are not among
 *  them. */
const std::vector<std::string> &dataset_preparation_keys();

/** The keys a potential file keeps only in its training record, in the
 *  order the record lists them: DBFILE (the training datasets), the keys
 *  of dataset_preparation_keys() (how the run prepared them), VALIDATION
 *  and HPOTARGET (the validation and target files of a tadah hpo run),
 *  UNCERTAINTY (the train task's switch for weights.tadah), VERBOSE,
 *  NUMERIC and OUTFILE (the run's verbosity and the writer's settings),
 *  AUDIT, AUDIT_SAMPLE and HEALTH_LOG (the run's diagnostics). The one
 *  list both rules read: no key line of these is written
 *  (remove_training_record_keys), and the record lists the run's values
 *  of these (training_record). */
const std::vector<std::string> &training_record_keys();

/** tadah predict's rule for the potential @p pot it predicts with (the
 *  context parsed from the file @p file): removes every key of
 *  dataset_preparation_keys() from @p pot -- they prepared the training
 *  data, not the data predicted, so a potential of an earlier version that
 *  has them as key lines predicts as one without them -- and returns ""
 *  when none of them had a value other than its blueprint default (a key at
 *  its default, such as the ESHIFT_ATOM false every potential of an earlier
 *  tadah train has, means the same read or ignored), else one line naming
 *  them with their values: "tadah predict: <file>: EFILTER -8 -2 and
 *  ZERO_COM_FORCE true prepared the training data of the potential and are
 *  not applied to the data predicted (tadah keeps such keys in a
 *  potential's training record, not as key lines); --efilter, --ffilter
 *  and --zero-com-force filter the data predicted.". */
std::string ignore_potential_dataset_keys(core::Context &pot,
                                          const std::string &file);

/** The rule of the parse of a run's configuration (TaskFactory parses
 *  every -c and --task file under it, a core::KeyLineRule): whether the
 *  line of @p key of a file whose key lines give @p file_keys is not read.
 *  A file that has WEIGHTS is a potential (or a copy of one): its lines of
 *  the keys of dataset_preparation_keys() prepared the data of the fit
 *  that wrote it, not of this run, and are not read -- ESHIFT_ATOM and
 *  ESHIFT_DBATOM only when the file has ESHIFT too (its ESHIFT, the energy
 *  reference of the model, overrules them in any configuration; without
 *  it a toggle is the only source of the shift, the way to derive ESHIFT
 *  from the data of a retrain, and is read). No other key, and no line of
 *  a file without WEIGHTS: the lines of the task file, of a configuration
 *  file that includes the potential with CONFIG (before or after the CONFIG
 *  line) and the command line are the run's own, so a retrain works with
 *  any number of datasets and with the run's own filters and weights,
 *  whatever version wrote the potential. */
bool potential_dataset_key_line(const std::set<std::string> &file_keys,
                                const std::string &key);

/** tadah train's and tadah hpo's notice for a configuration that is or
 *  includes a potential (TrainEngine::run, hpo_run_journaled), "" or a
 *  warning naming its lines of the keys that prepared the training data
 *  that were not read and are not at their blueprint default (a key at its
 *  default, such as the ZERO_COM_FORCE false every potential of an earlier
 *  tadah train has, means the same read or not), with the values; @p who is
 *  "tadah train" or "tadah hpo".
 *   - With @p provenance (a task of TaskFactory, whose parse did not read
 *     those lines: potential_dataset_key_line): @p config is left as it is;
 *     the lines are those of KeyProvenance::skipped_key_lines() whose key
 *     the run does not give itself (no line of its configuration or task
 *     file, no command-line option: a key it gives applies instead, and is
 *     not named), named per potential file:
 *     "<who>: the configuration file pot.train is a potential (it has
 *     WEIGHTS): its EFILTER 5 6 and WDBFILE 1 1 prepared the training data
 *     of the fit that wrote it, not of this run, and are ignored (tadah
 *     keeps such keys in a potential's training record, not as key lines).
 *     Give --efilter or --wdbfile on the command line, or an EFILTER or
 *     WDBFILE line in a configuration file that includes the potential with
 *     CONFIG, to apply one to this run (the Training page of the
 *     documentation, Retraining from a potential).", with "old.tadah, which
 *     the configuration file outer.train includes with CONFIG, is a
 *     potential ... or an EFILTER or WDBFILE line in outer.train, ..." for
 *     an included potential and "old.tadah, which the task file run.task
 *     includes with CONFIG, is a potential ... Write EFILTER or WDBFILE in
 *     the task file run.task to apply one to this run" for a task file (a
 *     run of a task file takes no command-line option); and, when a toggle
 *     is named, "The potential's ESHIFT, the energy reference of the model,
 *     is used: to derive ESHIFT from the data of this run, delete the
 *     ESHIFT line and set ESHIFT_ATOM or ESHIFT_DBATOM" as its last
 *     sentence.
 *   - Without @p provenance (a context a program built): @p config is taken
 *     as one file; when it has WEIGHTS, the keys potential_dataset_key_line
 *     selects by its keys are removed from it and named: "<who>: the
 *     configuration is a potential (it has WEIGHTS): its ... Give --efilter
 *     or --wdbfile on the command line to apply one to this run (...)".
 *  Call it before the record of the run is taken (training_record), so the
 *  record lists the run's own settings. */
std::string ignore_configuration_dataset_keys(
    core::Context &config, const KeyProvenance *provenance,
    const std::string &who);

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
 *  "# --- training record (no key is read from it) ---" (no loader of a
 *  potential reads a key from it; refit --edit reads the block to copy
 *  it). */
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
 *  the lines when the file cannot be read; when an opening line has no end
 *  line: another opening line, a line that is not a comment or the end of
 *  the file comes first (the refusal names the opening line and that
 *  line; every writer puts the key lines right after the record, so a
 *  potential whose end line was deleted is refused naming its first key
 *  line); when an end line has no opening line before it; and when a line
 *  inside a block, between its opening and end lines, is not a comment
 *  (every loader of the file reads it as a key). refit --edit copies its
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
