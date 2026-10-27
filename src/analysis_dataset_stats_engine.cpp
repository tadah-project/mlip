#include <tadah/mlip/detail/cell_geometry.h>
#include <tadah/mlip/engines/analysis_dataset_stats_engine.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/core/stats/data_stats_io.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>

namespace tadah {
namespace mlip {
namespace engines {

namespace {

// CLOSE_CONTACT when it is given, nullopt otherwise. A value that is not a
// finite distance greater than 0 is refused: no pair is closer than 0, so it
// could only report 0 pairs.
std::optional<double> close_contact(const tadah::core::Context& ctx) {
  if (!ctx.has_stored("CLOSE_CONTACT")) return std::nullopt;
  const double cc = ctx.get<double>("CLOSE_CONTACT");
  if (!detail::is_finite_bits(cc) || !(cc > 0.0))
    throw std::invalid_argument(
        "analysis dataset_stats: CLOSE_CONTACT (--close-contact) must be a "
        "finite distance greater than 0 (Angstrom), got " +
        tadah::core::format_roundtrip(cc));
  return cc;
}

// The cutoff of the statistics, RCUTMAX: the largest RCUT2B or RCUTMB value,
// which the parse of the configuration (-c) sets; --dbfile and --structure
// give none. A context without an RCUT2B or RCUTMB value, or whose largest
// is not above 0, is refused, before any data are read.
double stats_cutoff(const tadah::core::Context& ctx) {
  if (!ctx.has_stored("RCUT2B") && !ctx.has_stored("RCUTMB"))
    throw std::runtime_error(
        "analysis dataset_stats: no cutoff: the statistics take the largest "
        "RCUT2B or RCUTMB value of a configuration given with -c as their "
        "cutoff, and no line gives one (--dbfile and --structure give none)");
  const double rcut = ctx.get<double>("RCUTMAX");
  if (!(rcut > 0))
    throw std::runtime_error(
        "analysis dataset_stats: the cutoff of the statistics is the largest "
        "RCUT2B or RCUTMB value, " + tadah::core::format_roundtrip(rcut) +
        ", which is not above 0");
  return rcut;
}

// True when the command line gave `key` (--dbfile: DBFILE, --structure:
// STRUCTURE), as the task's KeyProvenance records it; false for a task
// without one (a Context a program builds).
bool on_command_line(const tadah::mlip::Task& task, const std::string& key) {
  const KeyProvenance* provenance = task.getKeyProvenance();
  return provenance != nullptr &&
         provenance->origin(key).kind == KeyOrigin::Kind::CommandLine;
}

}  // namespace

void AnalysisDatasetStatsEngine::validate(const tadah::mlip::Task& task) {
  // The command line refuses --dbfile and --structure without -c, and the
  // two together (the blueprint's cli_needs and cli_excludes). CLOSE_CONTACT
  // and the cutoff are checked here, before anything is read, computed or
  // written.
  close_contact(task.getContext());
  stats_cutoff(task.getContext());
}

int AnalysisDatasetStatsEngine::run(tadah::mlip::Task& task) {
  auto& taskContext = task.getContext();
  const std::optional<double> closeContact = close_contact(taskContext);
  stats_cutoff(taskContext);

  // Build a working context. If the user passed `-c train.tadah`, the
  // CLI dispatcher will have run CoreUserConfigParser on it already
  // and merged the result into taskContext (so RCUTMAX is set).
  tadah::core::Context workCtx = taskContext;

  // The data: the datasets of --dbfile or the structures of --structure
  // when the command line gives either, in place of the DBFILE and
  // STRUCTURE lines of the configuration, which are then not read (the
  // configuration gives the cutoff); else those lines. A task without a
  // KeyProvenance reads every DBFILE and STRUCTURE value of its context.
  const bool from_options = on_command_line(task, "DBFILE") ||
                            on_command_line(task, "STRUCTURE");
  const bool read_dbfile =
      taskContext.has_stored("DBFILE") &&
      (!from_options || on_command_line(task, "DBFILE"));
  const bool read_structure =
      taskContext.has_stored("STRUCTURE") &&
      (!from_options || on_command_line(task, "STRUCTURE"));

  // Collect structures.
  StructureDB stdb;
  if (read_dbfile) {
    const std::size_t nf = taskContext.size("DBFILE");
    for (std::size_t i = 0; i < nf; ++i) {
      const std::string dbf = taskContext.get<std::string>("DBFILE", i);
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::INFO, "Reading file: " + dbf);
      stdb.add(dbf);
      tadah::core::Logger::instance().log(
          tadah::core::LogLevel::INFO, stdb.summary());
    }
  }
  if (read_structure) {
    const std::size_t ns = taskContext.size("STRUCTURE");
    for (std::size_t i = 0; i < ns; ++i) {
      const std::string s = taskContext.get<std::string>("STRUCTURE", i);
      auto reader = StructureReaderSelector::getReader(s);
      reader->read(s);
      stdb.add(reader->getStructure());
    }
  }

  if (stdb.size() == 0) {
    // (A STRUCTURE value gives a structure, or its reader refuses it.)
    if (!read_dbfile)
      throw std::runtime_error(
          "analysis dataset_stats: no structures loaded: the configuration "
          "has no DBFILE or STRUCTURE line");
    std::string files;
    for (std::size_t i = 0; i < taskContext.size("DBFILE"); ++i)
      files += (i == 0 ? "" : " ") + taskContext.get<std::string>("DBFILE", i);
    throw std::runtime_error(
        "analysis dataset_stats: no structures loaded: the datasets of "
        "DBFILE (" + files + ") hold no structure");
  }

  // The audit of the data: two atoms at one position are a pair at
  // distance 0 (r_min 0), listed rather than refused.
  auto stats = compute_data_stats(stdb, workCtx, /*with_descriptor_stats=*/false,
                                  /*sample_size=*/0, /*list_coincident=*/true);

  // Pairs closer than CLOSE_CONTACT, from the lists compute_data_stats built
  // at RCUTMAX; a CLOSE_CONTACT beyond RCUTMAX is answered from lists out to
  // it, built for the scan only (find_close_contacts).
  std::vector<CloseContact> closePairs;
  if (closeContact) closePairs = find_close_contacts(stdb, *closeContact);

  // Structures whose cell is not symmetric (max |C - C^T| > 0). A cell that
  // an older Tadah stored transposed (see the data-migration note of the
  // 1.6.0 changelog) can only be told apart from a correct one in these, so
  // they are listed exactly; nothing is guessed about any of them.
  {
    std::vector<std::size_t> nonsym;
    for (std::size_t s = 0; s < stdb.size(); ++s) {
      const auto &C = stdb(s).cell;
      double d = 0.0;
      for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b)
          d = std::max(d, std::fabs(C(a, b) - C(b, a)));
      if (d > 0.0) nonsym.push_back(s);
    }
    std::cout << nonsym.size() << " of " << stdb.size()
              << " structures have a non-symmetric cell";
    if (nonsym.empty()) {
      std::cout << "." << std::endl;
    } else {
      std::cout << " (only in these can a cell stored transposed by an older "
                   "Tadah be told apart from a correct one; see the "
                   "data-migration note in the 1.6.0 changelog):"
                << std::endl;
      for (std::size_t s : nonsym)
        std::cout << "  structure " << s + 1 << ": " << stdb(s).label
                  << std::endl;
    }
  }

  // The close contacts, only when CLOSE_CONTACT is given: part of the
  // report asked for, so printed at every log level. Each pair once, in
  // dataset order, 1-based as `tadah data print` counts; the image shift
  // of the second atom when the pair is through a periodic image.
  if (closeContact) {
    std::cout << "Atom pairs closer than "
              << tadah::core::format_roundtrip(*closeContact)
              << " A (CLOSE_CONTACT): " << closePairs.size() << std::endl;
    for (const CloseContact& p : closePairs) {
      const Structure& st = stdb(p.structure);
      std::cout << "  structure " << p.structure + 1 << " ('" << st.label
                << "'): atoms " << p.i + 1 << " and " << p.j + 1 << " ("
                << st.atoms[p.i].symbol << ", " << st.atoms[p.j].symbol
                << "), distance " << tadah::core::format_sig(p.r, 10)
                << " A";
      if (p.shift[0] != 0 || p.shift[1] != 0 || p.shift[2] != 0)
        std::cout << ", image shift (" << p.shift[0] << ", " << p.shift[1]
                  << ", " << p.shift[2] << ")";
      std::cout << std::endl;
    }
  }

  // Output.
  std::string outpath = "dataset_stats.tadah";
  if (taskContext.contains("OUTFILE")) {
    outpath = taskContext.get<std::string>("OUTFILE");
  }
  std::ofstream os(outpath);
  if (!os) {
    throw std::runtime_error(
        "analysis dataset_stats: could not open output file '" + outpath + "'");
  }
  tadah::core::stats::save(stats, os);

  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      "analysis dataset_stats: wrote " + outpath +
          " (n_atoms=" + std::to_string(stats.n_atoms) +
          ", class=" + (stats.class_label.empty() ? "?" : stats.class_label) +
          ").");
  return EXIT_SUCCESS;
}

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
