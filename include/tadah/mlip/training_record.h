#ifndef TADAH_MLIP_TRAINING_RECORD_H
#define TADAH_MLIP_TRAINING_RECORD_H

#include <tadah/core/context.h>
#include <tadah/core/iuser_config_parser.h>  // core::SkippedKeyLine

#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

class KeyProvenance;

/** @file training_record.h
 *  The training record of a potential file:
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
 *  whose block is malformed.
 *
 *  A potential of an earlier version has some of these keys as key lines.
 *  Its lines of the keys of the record but DBFILE are not read by a later
 *  run (keys_not_read_from_a_potential, potential_line_not_read), and no
 *  parse stores its lines of the run settings among them
 *  (core::run_setting_keys(): VERBOSE, NUMERIC, OUTFILE, ...), so no
 *  program that loads it takes its verbosity or output from it. */

/** The keys that prepared the training data, in the order their
 *  transforms apply (apply_load_transforms), and WDBFILE: EFILTER and
 *  FFILTER (which structures were fitted), ESHIFT_ATOM and ESHIFT_DBATOM
 *  (how the run derived ESHIFT from its data: the values they produced are
 *  the potential's ESHIFT line), ZERO_COM_FORCE (the reference forces) and
 *  WDBFILE (the priority of each dataset in the fit). The retired
 *  WDBFILE_AUTO and EWEIGHT_TEMP, which a potential of an earlier version
 *  has as key lines, are not among them: no parse stores them (a fit
 *  refuses them, retired_keys.h). The model does not use them: a potential
 *  lists them in its training record (training_record_keys), tadah
 *  predict ignores them in a potential
 *  (ignore_potential_dataset_keys) and a retrain from a potential does not
 *  read the potential's lines of them (potential_line_not_read). LSCALE
 *  and ESHIFT, the model's length scale and energy reference, are not among
 *  them. */
const std::vector<std::string> &dataset_preparation_keys();

/** The keys whose lines in a potential a later run does not read
 *  (potential_line_not_read): every key of training_record_keys() but
 *  DBFILE, in that order -- those of dataset_preparation_keys() (they
 *  prepared the data of its own fit), VALIDATION, HPOTARGET, CHECK,
 *  VERBOSE, NUMERIC, OUTFILE, AUDIT, AUDIT_SAMPLE and HEALTH_LOG (settings
 *  of the run that wrote it: its validation and target files, the check
 *  instruction of tadah hpo, its verbosity, output and diagnostics; the run
 *  that reads the potential has its own). DBFILE is read: a retrain adds its
 *  DBFILE line to a copy of the potential. The run settings among them are
 *  core::run_setting_keys(), which no parse stores from a potential. */
const std::vector<std::string> &keys_not_read_from_a_potential();

/** The keys a potential file keeps only in its training record, in the
 *  order the record lists them: DBFILE (the training datasets), the keys
 *  of dataset_preparation_keys() (how the run prepared them), VALIDATION
 *  and HPOTARGET (the validation and target files of a tadah hpo run),
 *  CHECK (tadah hpo --check, which writes no potential: a potential of
 *  tadah train records the CHECK its configuration has, which tadah train
 *  does not read), VERBOSE, NUMERIC and OUTFILE (the run's verbosity and the writer's settings),
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
 *  every -c file under it, a core::KeyLineRule): whether the
 *  line of @p key of a file whose key lines give @p file_keys is not read.
 *  A file that has WEIGHTS is a potential (or a copy of one): its lines of
 *  the keys of keys_not_read_from_a_potential() are settings of the run
 *  that wrote it, not of this run, and are not read -- the keys that
 *  prepared the data of its fit (ESHIFT_ATOM and ESHIFT_DBATOM whether or
 *  not the file has ESHIFT), VALIDATION, HPOTARGET, CHECK, VERBOSE,
 *  NUMERIC, OUTFILE, AUDIT, AUDIT_SAMPLE and HEALTH_LOG. No other key
 *  (DBFILE is read: a retrain adds its DBFILE line to a copy of the
 *  potential), and no line of a file without WEIGHTS: the lines of a
 *  configuration file that includes the potential with CONFIG (before or
 *  after the CONFIG line) and the command line are the run's
 *  own, so a retrain works with any number of datasets and with the run's
 *  own filters, weights, ESHIFT toggle, verbosity, output and validation, whatever version wrote the potential. */
bool potential_line_not_read(const std::set<std::string> &file_keys,
                             const std::string &key);

/** The notice of tadah train, tadah hpo and tadah analysis check_bounds for
 *  a configuration that is or includes a potential (TrainEngine::run,
 *  hpo_run, AnalysisCheckBoundsEngine::run), "" or a
 *  warning naming its lines of keys_not_read_from_a_potential() that were
 *  not read, are not at their default (the blueprint task's default when
 *  it has one, else the key's: a key at its default, such as the
 *  ZERO_COM_FORCE false every potential of an earlier tadah train has, or
 *  the VERBOSE 1 of a best_pot.tadah of an earlier tadah hpo, means the
 *  same read or not) and are keys the run of the blueprint task @p task
 *  reads (the task as a user writes it, "train", "hpo" or "analysis
 *  check_bounds": the blueprint task "ANALYSIS.check_bounds"): the task's
 *  keys and those of its input groups (tadah hpo's VALIDATION and
 *  HPOTARGET), and, for tadah train and tadah hpo, NUMERIC, AUDIT,
 *  AUDIT_SAMPLE and, for tadah hpo, HEALTH_LOG, which both read from any
 *  configuration (tadah hpo takes no OUTFILE, tadah train no
 *  VALIDATION, HPOTARGET or HEALTH_LOG, check_bounds only HPOTARGET and
 *  VERBOSE: such a line is not named); with the values; "tadah <task>"
 *  begins it.
 *   - With @p provenance (a task of TaskFactory, whose parse did not read
 *     those lines: potential_line_not_read): @p config is left as it is;
 *     the lines are those of KeyProvenance::skipped_key_lines() whose key
 *     the run does not give itself (no line of its configuration file, no
 *     command-line option: a key it gives applies instead, and is
 *     not named) -- nor, for a line of ESHIFT_ATOM or ESHIFT_DBATOM, the
 *     other toggle set true (the two choose one thing, the source of the
 *     shift, and exclude each other: the run's own toggle is its choice) --
 *     named per potential file:
 *     "tadah train: the configuration file pot.train is a potential (it
 *     has WEIGHTS): its EFILTER 5 6 and WDBFILE 1 1 prepared the training
 *     data of the fit that wrote it, not of this run, and are ignored
 *     (tadah keeps such keys in a potential's training record, not as key
 *     lines). This run takes its own EFILTER or WDBFILE from --efilter or
 *     --wdbfile on the command line, or from an EFILTER or WDBFILE line in
 *     a configuration file that includes the potential with CONFIG (the
 *     Training page of the documentation, Retraining from a potential).",
 *     with "old.tadah, which the configuration file outer.train includes
 *     with CONFIG, is a potential ... or from an EFILTER or WDBFILE line in
 *     outer.train ..." for an included potential. The run settings
 *     are named in a sentence of their own, "Its VERBOSE 2 and NUMERIC 12 were settings
 *     of the run that wrote it, not of this run, and are ignored too." (or
 *     first: "its OUTFILE old.tadah was a setting of the run that wrote
 *     it, not of this run, and is ignored (tadah keeps ...).");
 *     the advice names every key named, ESHIFT_ATOM, ESHIFT_DBATOM and the
 *     run settings among them, and the options of
 *     those the task has (tadah train has no --numeric: "This run takes
 *     its own VERBOSE or NUMERIC from --verbose on the command line, or
 *     from a VERBOSE or NUMERIC line in ..."); and, when a toggle is named
 *     and @p config has ESHIFT, its last sentence says that the ESHIFT
 *     overrules the toggles and when ESHIFT is derived from the data of
 *     the run, by where the ESHIFT came from: "The ESHIFT of the
 *     configuration, the energy reference of the model, overrules
 *     ESHIFT_ATOM and ESHIFT_DBATOM: ESHIFT is derived from the data of
 *     this run when the ESHIFT line is deleted and the run is given one of
 *     them" for a line of a file, "The ESHIFT given with --eshift, ...:
 *     ESHIFT is derived from the data of this run when --eshift is left
 *     out and the run is given one of them" for the option, and "... when
 *     --eshift is left out, the ESHIFT line is deleted and the run is given
 *     one of them" when the option replaced a line of the configuration.
 *   - Without @p provenance (a context a program built): @p config is taken
 *     as one file; when it has WEIGHTS, the keys potential_line_not_read
 *     selects by its keys that a parse stores -- those of
 *     dataset_preparation_keys() -- are removed from it and
 *     named: "tadah train: the configuration is a potential (it has
 *     WEIGHTS): its ... This run takes its own EFILTER or WDBFILE from
 *     --efilter or --wdbfile on the command line (...)". Its run settings
 *     (VERBOSE, NUMERIC,
 *     OUTFILE, ...: core::run_setting_keys()) are kept: no parse stores a
 *     potential's, so they are the program's own.
 *  Call it before the record of the run is taken (training_record), so the
 *  record lists the run's own settings. */
std::string ignore_configuration_potential_lines(
    core::Context &config, const KeyProvenance *provenance,
    const std::string &task);

/** The refusal of a run of @p task (as ignore_configuration_potential_lines
 *  takes it: "hpo", "analysis check_bounds") whose configuration has no
 *  @p key, a setting the run needs that names @p what ("the target
 *  file"); "tadah <task>: no <key>: " begins it. It says how the run's own
 *  is given (tadah analysis check_bounds -c, a program's context for
 *  hpo_run; the command line gives tadah hpo -c an --hpotarget): "the
 *  target file is given with --hpotarget, or with an HPOTARGET line in a
 *  configuration file without WEIGHTS." (the option only when the task has
 *  one). Each potential file whose line of @p key its parse did not read
 *  (KeyProvenance::skipped_key_lines of @p provenance; nullptr: a context a
 *  program built) adds a sentence: "old.tadah is a potential (it has
 *  WEIGHTS): its HPOTARGET config.hpo line was a setting of the run that
 *  wrote it and is not read." */
std::string missing_run_setting(const std::string &task,
                                const std::string &key,
                                const std::string &what,
                                const KeyProvenance *provenance);

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

/** The training record of the key lines of an older potential, which
 *  refit --edit adds to the record it copies: one "KEY v1 v2 ..." line for
 *  each key of training_record_keys() in that order, with the tokens
 *  @p pot stores (its DBFILE: a parse reads it) or, for a key @p pot does
 *  not store, the values of its lines in @p not_read (the lines the parse
 *  of the potential did not read: potential_line_not_read and the run
 *  settings no parse stores; the values of several lines of one key joined
 *  in order). Empty for a potential tadah wrote (it has no such line). */
TrainingRecord record_of_key_lines(
    const core::Context &pot,
    const std::vector<core::SkippedKeyLine> &not_read);

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
 *  copy of @p pot without the keys of training_record_keys(), then, as
 *  they were written, the lines of retired keys @p pot records
 *  (core::Context::retired_keys(): refit --edit of a potential of an
 *  earlier version keeps them, so a fit of the edited potential refuses
 *  them as it refuses the input; a fit writes none, it refuses them). The body of
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
