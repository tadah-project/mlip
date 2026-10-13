#ifndef TADAH_MLIP_TRAINING_DATASETS_H
#define TADAH_MLIP_TRAINING_DATASETS_H

#include <tadah/core/context.h>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tadah {
namespace mlip {

/** What a potential records: the model's settings and its fit, not the
 *  data it was fitted to. Removes DBFILE, the training datasets, from the
 *  potential context @p pot; every other key stays (ALPHA and BETA, the
 *  evidence approximation's starting guesses or the values a LAMBDA -1 fit
 *  found, WDBFILE, the weight of each dataset, ...). M_BLR::get_param_file
 *  and M_KRR::get_param_file (the potential of a fit: tadah train, tadah
 *  hpo) and TrainEngine::dumpPotFile (the files of tadah train, tadah
 *  refit and refit --edit, whose potential is a copy of its input's) call
 *  it, so no potential tadah writes has DBFILE, whichever the model. A
 *  potential retrains as a configuration with the DBFILE of the retrain
 *  added (`tadah train -c pot.tadah` plus a DBFILE line), and its own
 *  dataset is never listed twice. */
void remove_training_datasets(tadah::core::Context &pot);

/** What a potential records: the model's settings and its fit, not these
 *  inputs of the run that wrote it. Removes from the potential context
 *  @p pot VERBOSE (the verbosity of the tool), NUMERIC and OUTFILE (the
 *  writer's settings; the digits are resolved before), AUDIT, AUDIT_SAMPLE
 *  and HEALTH_LOG (the run's diagnostics), HPOTARGET and VALIDATION (the
 *  target and validation files of a tadah hpo run, which the run's journal
 *  names), and UNCERTAINTY (the train task's switch for weights.tadah,
 *  which M_BLR's and M_KRR's get_param_file leave out too: refit --edit
 *  writes a copy of its input's context, which a potential of tadah 1.5.x
 *  or a 1.6.0 development build gives an UNCERTAINTY line); every other
 *  key stays. TrainEngine::dumpPotFile (tadah
 *  train, tadah refit and refit --edit) and Tadah.HPO's
 *  PotHeaderWriter::write_potential_body (best_pot.tadah, the durable best,
 *  the BDUMP and DUMP archives) call it on the context they write, so no
 *  potential file tadah writes has these keys; the context of a tadah hpo
 *  evaluation keeps them, and the models the run builds from it (the
 *  PC_LAMMPS potential among them) take the run's VERBOSE. A potential that
 *  has them (written before) still loads. tadah predict and pair_style
 *  tadah read none of them except VERBOSE: parsing a VERBOSE line sets the
 *  Tadah log level of the process (core::Context::add), and the models
 *  built from the file take its verbosity. */
void remove_run_inputs(tadah::core::Context &pot);

/** The advice the refusals of a configuration without a training DBFILE
 *  end with (require_training_datasets: tadah train; classify_weight_source
 *  and the refusal of a run without ATOM: tadah hpo): "A potential records
 *  no DBFILE (no potential tadah writes has one): to train from one, add a
 *  DBFILE line to a copy of it; a potential of tadah refit or refit --edit
 *  also pins every weight with its FIXINDEX and FIXWEIGHT lines, the seed
 *  of a tadah hpo coefficient search, so delete them from the copy for the
 *  fit to change the weights (the Training page of the documentation,
 *  Retraining from a potential)." A fit whose FIXINDEX pins every weight
 *  changes none (all_weights_pinned_warning, weight_provisioning.h). */
std::string potential_has_no_datasets_advice();

/** Refuses a training configuration without DBFILE (tadah train, which
 *  fits the datasets it lists and has no other source of them): throws
 *  std::runtime_error "The training configuration has no DBFILE: tadah
 *  train fits the datasets it lists. " followed by
 *  potential_has_no_datasets_advice(), so `tadah train -c pot.tadah`
 *  without a DBFILE line is told how to retrain. A no-op with DBFILE.
 *  TrainEngine::run calls it before any file is read. */
void require_training_datasets(const tadah::core::Context &ctx);

/** The first two entries of the dataset list @p files that name one file,
 *  as their 0-based indices (i, j), i < j, j the smallest such, or
 *  std::nullopt when every entry names a file of its own. Two entries are
 *  one file when std::filesystem::equivalent says so (two spellings of a
 *  path, a symbolic link), or, for a file that does not exist, when their
 *  absolute normal paths are equal. The rule of
 *  require_distinct_training_datasets, and of tadah hpo for the datasets
 *  of its validation file. */
std::optional<std::pair<std::size_t, std::size_t>> repeated_dataset(
    const std::vector<std::string> &files);

/** Refuses a training configuration whose DBFILE lists one dataset file
 *  twice: StructureDB::add loads every DBFILE entry as a dataset, so each
 *  structure of the file would be fitted twice (a fit of twice the rows,
 *  LAMBDA at half strength, an OALGO 3 covariance of half the size). Two
 *  entries are one file by the rule of repeated_dataset. Throws
 *  std::runtime_error naming both entries, WDBFILE (the weight of a
 *  dataset) and the M_BLR potentials that recorded their DBFILE; a no-op
 *  without DBFILE. tadah train (TrainEngine::run, before any file is read)
 *  and tadah hpo (validate_hpo_setup, at startup) call it. */
void require_distinct_training_datasets(const tadah::core::Context &ctx);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_TRAINING_DATASETS_H
