#ifndef TADAH_MLIP_ENGINES_ANALYSIS_DATASET_STATS_ENGINE_H
#define TADAH_MLIP_ENGINES_ANALYSIS_DATASET_STATS_ENGINE_H

#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

/**
 * @brief Computes dataset statistics and writes them to the file OUTFILE
 * (--outfile) names, a report: tadah analysis suggest_bounds, check_bounds
 * and the pre-flight audit compute the statistics they need themselves.
 *
 * Reads DBFILEs (and optionally individual STRUCTUREs) from the task
 * context, computes per-element-pair distance percentiles + KDE peaks,
 * neighbour-count percentiles per element, energy/force scales, and a
 * material-class label.
 * The cutoff is RCUTMAX, the largest cutoff of the configuration (-c): its
 * RCUT2B and RCUTMB values and the cutoff RADIUS of each DESCRIPTOR block
 * (resolved with models::resolve_embed_blocks, as tadah train resolves
 * them). The data are its DBFILE and STRUCTURE lines, or, when
 * the command line gives --dbfile or --structure (KeyProvenance:
 * KeyOrigin::Kind::CommandLine), the values of those options alone: the
 * configuration's DBFILE and STRUCTURE lines are then not read. A task
 * without a KeyProvenance reads every DBFILE and STRUCTURE value of its
 * context.
 * Every pair strictly inside RCUTMAX counts, a pair at distance 0 (two
 * atoms at one position) included, so r_min is 0 for such a dataset
 * (compute_data_stats).
 *
 * Prints to standard output, at every log level, how many structures have
 * a non-symmetric cell and lists them ("structure <i>: <label>"). With
 * CLOSE_CONTACT (--close-contact <A>) it also prints
 * "Atom pairs closer than <A> A (CLOSE_CONTACT): <K>" and one line per
 * pair closer than A, each unordered pair once (a pair through a periodic
 * image included), in dataset order (find_close_contacts):
 * "structure <s> ('<label>'): atoms <i> and <j> (<Si>, <Sj>), distance
 * <r> A", followed by ", image shift (<n0>, <n1>, <n2>)" (of atom j) for a
 * pair through an image; s, i and j are 1-based. A CLOSE_CONTACT beyond
 * RCUTMAX is answered in full (lists out to it are built for the scan).
 * validate() refuses a CLOSE_CONTACT that is not a finite distance greater
 * than 0, and a context without an RCUT2B or RCUTMB value or a DESCRIPTOR
 * block, or whose RCUTMAX is not above 0, before anything is read, computed
 * or written; run()
 * refuses them likewise before it reads the data.
 */
class AnalysisDatasetStatsEngine: public IEngine {
public:
  int run(tadah::mlip::Task &task) override;
  void validate(const tadah::mlip::Task &task) override;
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif  // TADAH_MLIP_ENGINES_ANALYSIS_DATASET_STATS_ENGINE_H
