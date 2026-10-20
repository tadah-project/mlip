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

/** The model's settings and its fit, not the data it was fitted to:
 *  removes DBFILE, the training datasets, from the potential context
 *  @p pot; every other key stays (ALPHA and BETA, the evidence
 *  approximation's starting guesses or the values a LAMBDA -1 fit found,
 *  WDBFILE, the weight of each dataset, ...). M_BLR::get_param_file and
 *  M_KRR::get_param_file (the potential of a fit: tadah train, tadah hpo)
 *  call it, so the context of a fitted potential -- the potential of a
 *  tadah hpo evaluation, which its PC_LAMMPS scripts run -- has no DBFILE,
 *  whichever the model. No potential file has a DBFILE key line either
 *  (write_potential_with_record, training_record.h): its training record,
 *  a comment block no loader of a potential reads, lists the datasets of
 *  its fit when there were any. A potential retrains as a configuration with the
 *  DBFILE of the retrain added (`tadah train -c pot.tadah` plus a DBFILE
 *  line), and its own dataset is never listed twice. */
void remove_training_datasets(tadah::core::Context &pot);

/** The advice the refusals of a configuration without a training DBFILE
 *  end with (require_training_datasets: tadah train; classify_weight_source
 *  and the refusal of a run without ATOM: tadah hpo): "A potential has no
 *  DBFILE line (no potential tadah writes one; its training record, the
 *  comment block above its key lines, lists the datasets its weights were
 *  fitted to in a '# DBFILE' line when there were any: a refit has none,
 *  and a potential of an earlier version has no record): to train from
 *  one, add a DBFILE line to a copy of it; a potential of tadah refit or
 *  refit --edit also pins every weight with its FIXINDEX and FIXWEIGHT
 *  lines, the seed of a tadah hpo coefficient search, so delete them from
 *  the copy for the fit to change the weights (the Training page of the
 *  documentation, Retraining from a potential)." A fit
 *  whose FIXINDEX pins every weight changes none
 *  (all_weights_pinned_warning, weight_provisioning.h). */
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
 *  LAMBDA at half strength). Two
 *  entries are one file by the rule of repeated_dataset. Throws
 *  std::runtime_error naming both entries, WDBFILE (the weight of a
 *  dataset) and the M_BLR potentials that recorded their DBFILE; a no-op
 *  without DBFILE. tadah train (TrainEngine::run, before any file is read)
 *  and tadah hpo (validate_hpo_setup, at startup) call it. */
void require_distinct_training_datasets(const tadah::core::Context &ctx);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_TRAINING_DATASETS_H
