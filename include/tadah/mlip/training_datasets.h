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

/** What a potential records: the model's settings and its fit, not the
 *  inputs of the run that wrote it. Removes from the potential context
 *  @p pot VERBOSE (the verbosity of the tool), NUMERIC and OUTFILE (the
 *  writer's settings; the digits are resolved before), AUDIT, AUDIT_SAMPLE
 *  and HEALTH_LOG (the run's diagnostics), and HPOTARGET and VALIDATION
 *  (the target and validation files of a tadah hpo run, which the run's
 *  journal names). TrainEngine::dumpPotFile (tadah train, tadah refit and
 *  refit --edit) and tadah hpo's potential of an evaluation (best_pot.tadah,
 *  the BDUMP and DUMP archives, the potential its PC_LAMMPS scripts run)
 *  call it, so no potential tadah writes has these keys. A potential that
 *  has them (written before) still loads: they are configuration keys
 *  that tadah predict and pair_style tadah do not read. */
void remove_run_inputs(tadah::core::Context &pot);

/** The advice every refusal of a configuration without a training DBFILE
 *  ends with (tadah train, tadah hpo): "A potential records no DBFILE (no
 *  potential tadah writes has one): to train from one, add a DBFILE line
 *  to a copy of it (the Training page of the documentation, Retraining from
 *  a potential)." */
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
