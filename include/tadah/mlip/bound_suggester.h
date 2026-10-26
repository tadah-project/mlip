#ifndef TADAH_MLIP_BOUND_SUGGESTER_H
#define TADAH_MLIP_BOUND_SUGGESTER_H

// `tadah analysis suggest_bounds`: OPTIM bound proposals for every
// parameter declared in each descriptor's meta(), from the data and the
// seed. Part of the HPO bound-validation pipeline.
//
// The suggester reads dataset statistics (a `DataStats` produced by
// compute_data_stats) plus the seed `Context` of the training
// configuration, walks the configured descriptor's metadata, and returns
// one `OptimSuggestion` per (key, arg_pos) pair. Each suggestion carries
// its rationale: the form it assumes and the statistics it comes from, or
// that its range is a fixed one, or why no OPTIM line is suggested.

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
  bool         fixed{false};         ///< a fixed range, not from the data
  bool         none{false};          ///< no OPTIM line; rationale says why
  models::Criticality crit{models::Criticality::Optional};
  models::Scale       scale{models::Scale::Dimensionless};
  std::string  units{};
};

/** \brief Suggest OPTIM bounds for one descriptor's meta().
 *
 *  \param data        DataStats produced by compute_data_stats.
 *  \param meta        DescriptorMeta returned by the descriptor's
 *                     `D_Base::meta()`.
 *  \param seed        Context with seed values (the training
 *                     configuration, its radial grids expanded:
 *                     D_mJoin::expand_grids).
 *  \return vector of OptimSuggestion, in the order params appear in
 *          `meta.params`. For vector-valued params (arg_pos >= 0)
 *          the suggester is called once per existing seed entry. A
 *          suggestion with `none` set has no range (no OPTIM line); one
 *          with `fixed` set has a range not derived from the data. Each
 *          seed value is read as a number (a value that is none throws).
 */
std::vector<OptimSuggestion>
suggest_bounds_for(const tadah::core::stats::DataStats& data,
                   const models::DescriptorMeta&        meta,
                   const tadah::core::Context&          seed);

/** \brief Pretty-print suggestions as a human table: one row per
 *  suggestion, its rationale saying whether its range is a fixed one, and
 *  one row "no OPTIM line" per suggestion without a range. Leaves the
 *  format of @p os as it found it. */
void print_suggestions(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os);

/** \brief Write suggestions as a compilable `OPTIM ...` block. The
 *  output is appendable to an HPO target file. The entries of a vector
 *  key that follow one another with the same range share one line
 *  "(i-j)"; an entry without a range gets a comment line instead. Leaves
 *  the format of @p os as it found it. */
void write_optim_block(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_BOUND_SUGGESTER_H
