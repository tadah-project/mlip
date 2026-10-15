#ifndef TADAH_MLIP_BOUND_SUGGESTER_H
#define TADAH_MLIP_BOUND_SUGGESTER_H

// `tadah hpo --suggest-bounds` — produce data-driven OPTIM bound
// proposals for every parameter declared in each descriptor's meta().
// Part of the HPO bound-validation pipeline.
//
// The suggester reads dataset statistics (a `DataStats` produced by
// compute_data_stats) plus the seed `Context` from the user's config,
// walks the configured descriptor's metadata, and returns one
// `OptimSuggestion` per (key, arg_pos) pair. Each suggestion carries
// a human-readable rationale so the user can review and override.

#include <iosfwd>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/models/descriptors/param_meta.h>

namespace tadah {
namespace mlip {

struct OptimSuggestion {
  std::string  descriptor_label{};   ///< owning descriptor (e.g. "DM_mBlipEnv")
  std::string  key{};                ///< OPTIM key name
  int          arg_pos{-1};          ///< -1 scalar; 0..n-1 vector entry
  std::optional<std::pair<std::string, std::string>> pair{};
  double       lo{0.0};
  double       hi{0.0};
  std::optional<double> seed_value{};
  std::string  rationale{};          ///< human-readable explanation
  models::Criticality crit{models::Criticality::Optional};
  models::Scale       scale{models::Scale::Dimensionless};
  std::string  units{};
};

/** \brief Suggest OPTIM bounds for one descriptor's meta().
 *
 *  \param data        DataStats produced by compute_data_stats.
 *  \param meta        DescriptorMeta returned by the descriptor's
 *                     `D_Base::meta()`.
 *  \param seed        Context with seed values (the user's
 *                     `train.tadah` after CoreUserConfigParser
 *                     postprocessing).
 *  \return vector of OptimSuggestion, in the order params appear in
 *          `meta.params`. For vector-valued params (arg_pos >= 0)
 *          the suggester is called once per existing seed entry.
 */
std::vector<OptimSuggestion>
suggest_bounds_for(const tadah::core::stats::DataStats& data,
                   const models::DescriptorMeta&        meta,
                   const tadah::core::Context&          seed);

/** \brief Pretty-print suggestions as a human table. */
void print_suggestions(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os);

/** \brief Write suggestions as a compilable `OPTIM ...` block. The
 *  output is appendable to an HPO target file. */
void write_optim_block(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_BOUND_SUGGESTER_H
