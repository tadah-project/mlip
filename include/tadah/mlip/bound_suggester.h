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
// its rationale: the form it assumes and the statistics it comes from,
// that its range is a fixed one, or, from a suggester that gives none, the
// scale and the role of the parameter; or why no OPTIM line is suggested.
// A range holds the seed's value, where tadah hpo starts: one widened to
// hold it says so. A range of zero width, which tadah hpo refuses, is no
// range: such a row has no OPTIM line. The neighbour statistics reach past
// the seed's cutoffs (models::kCutoffSearchReach times the largest).

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
 *          `meta.params`. For vector-valued params (arg_pos 0) the
 *          suggester is called once per existing seed entry. A value of
 *          the descriptor's TYPE2B or TYPEMB line (ParamMeta arg_pos > 0,
 *          its position on the descriptor's line) gets a suggestion with
 *          `none` set, as tadah hpo searches no value of those lines, its
 *          seed value read from the descriptor's line and its entry
 *          numbered as an OPTIM line numbers the values of the key, by the
 *          numbers among them (a0 of "D2_ZBL a0 s0 p0 p1 Ta Ta" is entry
 *          1); without the line in @p seed it has no row, and a line that
 *          is not the descriptor's throws std::logic_error. A suggestion
 *          with `none` set has no range (no OPTIM line); one with `fixed`
 *          set has a range not derived from the data. A range that does
 *          not hold the seed's value is widened to it (its rationale says
 *          so): tadah hpo refuses a starting point outside its box. A
 *          range of zero width after that gets `none`, its rationale
 *          saying so: tadah hpo refuses such a box. Each seed value is
 *          read as a number (a value that is none throws). For a
 *          descriptor of a joined line, @p seed is its own context
 *          (joined_descriptors()), and the caller numbers its entries as
 *          the whole key does.
 *  \param pairs       The element pairs of the descriptor's TYPE2B or
 *                     TYPEMB line (D_Base::get_init_atoms()): a descriptor
 *                     sums over the neighbours of those pairs, so each
 *                     range is taken from their statistics
 *                     (LinePairs::stats()), the suggester called on those
 *                     of each pair, and the range holds the range of each,
 *                     or, for a parameter marked
 *                     models::ParamMeta::reach_each_pair (a cutoff), runs
 *                     from the largest low end of the pairs' ranges to the
 *                     largest high end, so that each value in it reaches
 *                     each pair, as the check of the cutoff judges it (a
 *                     rationale says which and gives each pair's); a row
 *                     of data without such a pair names them. Without them
 *                     (the metadata of no line), from @p data as given.
 */
std::vector<OptimSuggestion>
suggest_bounds_for(const tadah::core::stats::DataStats& data,
                   const models::DescriptorMeta&        meta,
                   const tadah::core::Context&          seed,
                   const std::optional<models::LinePairs>& pairs =
                       std::nullopt);

/** \brief One row per key and entry, as tadah hpo takes one OPTIM line
 *  for an entry: the rows of @p rows with the key and entry of an earlier
 *  row (the descriptors of a joined line give one each for a key they
 *  share, RCUTENV of two DM_mBlipEnv) are merged into the earlier one, in
 *  place: its range widened to hold each range, its descriptor label
 *  naming each descriptor once ("DM_mBlipEnv", "DM_A+DM_B"), and its
 *  rationale saying that the descriptors share the key (and giving the
 *  rationale of a row that differs). Entry 1 of a key of one value
 *  (arg_pos -1) is entry 1 of the key. */
void merge_shared_rows(std::vector<OptimSuggestion>& rows);

/** \brief The rows of a cutoff that tadah hpo derives from the largest
 *  knot get no OPTIM line: RCUT2B of a D2_Knot5 line alone on the TYPE2B
 *  line of @p seed, and RCUTMB of a knot DM_REAM density alone on its
 *  TYPEMB line (models::potedit::derived_knot_rcut; @p seed is the whole
 *  training configuration), and, on a D2_mJoin or DM_mJoin line, the RCUT
 *  entry of each such descriptor (joined_descriptors(): its own context
 *  and entry). tadah train and tadah hpo take the largest knot as the
 *  cutoff of such a descriptor, and tadah hpo warns that an OPTIM line of
 *  it has no effect; each such row of @p rows gets `none`, its rationale
 *  naming the largest knot. */
void leave_out_derived_cutoffs(std::vector<OptimSuggestion>& rows,
                               const tadah::core::Context& seed);

/** \brief Pretty-print suggestions as a human table: one row per
 *  suggestion, named "descriptor.key(i)" with i the entry as an OPTIM line
 *  numbers it (from 1), with its rationale (the form and the data, "fixed
 *  range" and why, or the scale and the role of the parameter), and one
 *  row "no OPTIM line" per suggestion without a range. Leaves the format
 *  of @p os as it found it. */
void print_suggestions(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os);

/** \brief Write suggestions as a compilable `OPTIM ...` block. The
 *  output is appendable to an HPO target file. The entries of a vector
 *  key that follow one another with the same range share one line
 *  "(i-j)"; an entry without a range gets a comment line instead. A bound
 *  is written with 7 significant digits, the low one rounded down and the
 *  high one up, so that the box holds its range (and a seed's value at an
 *  end of it). Leaves the format of @p os as it found it. */
void write_optim_block(const std::vector<OptimSuggestion>& sug,
                       std::ostream& os);

}  // namespace mlip
}  // namespace tadah

#endif  // TADAH_MLIP_BOUND_SUGGESTER_H
