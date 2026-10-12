#ifndef TADAH_MLIP_TRAINING_DATASETS_H
#define TADAH_MLIP_TRAINING_DATASETS_H

#include <tadah/core/context.h>

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

/** Refuses a training configuration whose DBFILE lists one dataset file
 *  twice: StructureDB::add loads every DBFILE entry as a dataset, so each
 *  structure of the file would be fitted twice (a fit of twice the rows,
 *  LAMBDA at half strength, an OALGO 3 covariance of half the size). Two
 *  entries are one file when std::filesystem::equivalent says so (two
 *  spellings of a path, a symbolic link), or, for a file that does not
 *  exist, when their absolute normal paths are equal. Throws
 *  std::runtime_error naming both entries, WDBFILE (the weight of a
 *  dataset) and the M_BLR potentials that recorded their DBFILE; a no-op
 *  without DBFILE. tadah train (TrainEngine::run, before any file is read)
 *  and tadah hpo (validate_hpo_setup, at startup) call it. */
void require_distinct_training_datasets(const tadah::core::Context &ctx);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_TRAINING_DATASETS_H
