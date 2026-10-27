#include <tadah/mlip/engines/analysis_suggest_bounds_engine.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/core/registry.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/bound_suggester.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/joined_descriptors.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/models/descriptors/density_reach.h>
#include <tadah/models/descriptors/d2/d2_base.h>
#include <tadah/models/descriptors/dm/dm_base.h>

namespace tadah {
namespace mlip {
namespace engines {

void AnalysisSuggestBoundsEngine::validate(const tadah::mlip::Task& /*task*/) {
  // Blueprint validation enforces input-group constraints (`-c | -t`).
  // Engine-level pre-conditions (e.g. RCUTMAX presence) are checked
  // inside run() with actionable error messages.
}

int AnalysisSuggestBoundsEngine::run(tadah::mlip::Task& task) {
  auto& taskCtx = task.getContext();
  tadah::core::Context workCtx = taskCtx;

  if (!taskCtx.contains("DBFILE")) {
    throw std::runtime_error(
        "analysis suggest_bounds: at least one DBFILE must be configured "
        "(via `-c train.tadah`).");
  }
  if (!workCtx.contains("RCUTMAX") || workCtx.get<double>("RCUTMAX") <= 0) {
    throw std::runtime_error(
        "analysis suggest_bounds: RCUTMAX is not set (set RCUT2B / RCUTMB).");
  }

  // Load DB
  StructureDB stdb;
  const std::size_t nf = taskCtx.size("DBFILE");
  for (std::size_t i = 0; i < nf; ++i) {
    const std::string dbf = taskCtx.get<std::string>("DBFILE", i);
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::INFO, "Reading file: " + dbf);
    stdb.add(dbf);
  }
  if (stdb.size() == 0) {
    throw std::runtime_error("analysis suggest_bounds: no structures loaded");
  }

  // The radial grids as the descriptors read them: a grid generator
  // ("CGRID2B LIN 7 0.1 3.0") expanded into its values, as tadah train and
  // tadah hpo expand it, so that the suggestions have one row per value
  // and read the seed's values as numbers.
  models::D_mJoin::expand_grids(workCtx);

  // Compute dataset stats (DBFILE -> DataStats). No descriptor
  // statistics: the ranges that need them (ρ) are fixed ones, which their
  // rationale says. The neighbour distances are taken up to
  // models::kCutoffSearchReach times the seed's cutoffs (RCUTMAX follows
  // RCUT2B, RCUTMB and the cutoff radius of each DESCRIPTOR block, which
  // expand_grids has resolved), so that the range of a cutoff can end past
  // the seed's: statistics within the seed's cutoff hold no distance beyond
  // it.
  tadah::core::Context statsCtx = workCtx;
  for (const char* key : {"RCUT2B", "RCUTMB"}) {
    if (!statsCtx.has_stored(key)) continue;
    std::vector<double> values(statsCtx.size(key));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = models::kCutoffSearchReach * statsCtx.get<double>(key, i);
    statsCtx.remove(key);
    for (double v : values)
      statsCtx.add(key, tadah::core::format_roundtrip(v));
  }
  std::vector<double> radii;
  for (std::size_t i = 0;; ++i) {
    const auto radius = statsCtx.stored_value("COMPONENT_RCUTMB", i);
    if (!radius) break;
    radii.push_back(models::kCutoffSearchReach *
                    tadah::core::to_double(*radius));
  }
  if (!radii.empty()) statsCtx.set_component_cutoffs(radii);
  tadah::core::Logger::instance().log(
      tadah::core::LogLevel::INFO,
      "analysis suggest_bounds: the neighbour statistics reach " +
          tadah::core::format_roundtrip(statsCtx.get<double>("RCUTMAX")) +
          " Å, " + tadah::core::format_roundtrip(models::kCutoffSearchReach) +
          " times the largest cutoff of the seed (" +
          tadah::core::format_roundtrip(workCtx.get<double>("RCUTMAX")) +
          " Å)");
  auto data =
      compute_data_stats(stdb, statsCtx, /*with_descriptor_stats=*/false);

  // Output stream — stdout if no OUTFILE, else file.
  std::ofstream ofile;
  std::ostream* os_block = &std::cout;
  if (taskCtx.contains("OUTFILE")) {
    const std::string out = taskCtx.get<std::string>("OUTFILE");
    ofile.open(out);
    if (!ofile) {
      throw std::runtime_error(
          "analysis suggest_bounds: cannot open output file '" + out + "'");
    }
    os_block = &ofile;
  }

  // Configured descriptors (string names from the user's TYPE2B/TYPEMB).
  std::vector<OptimSuggestion> all_suggestions;

  // A line of one descriptor: the entry of TYPE2B / TYPEMB that starts
  // with `D2_` or `DM_` (its other entries are its parameters and
  // elements), read from the configuration. A joined line (`D2_mJoin`,
  // `DM_mJoin`) gives each of its descriptors its own values.
  auto looks_like_descriptor = [](const std::string& s) {
    return s.size() >= 3 && (s.compare(0, 3, "D2_") == 0 ||
                             s.compare(0, 3, "DM_") == 0);
  };

  auto run_for = [&](const std::string& type_key) {
    if (!workCtx.contains(type_key)) return;
    // A joined line (D2_mJoin, DM_mJoin), and the descriptors a TYPEMB key
    // places among DESCRIPTOR blocks (block_line_descriptors(); the
    // blocks' rows come below): each descriptor's rows from its own values
    // (its RCUT value and its share of each list key) and the element
    // pairs of its own line, each numbered by the place of its value in
    // the whole key, as an OPTIM line numbers it (the RCUT row of the k-th
    // descriptor is entry k). A key the line splits but the descriptor
    // takes no value of has no row.
    const bool blocks =
        type_key == "TYPEMB" && !workCtx.blocks("DESCRIPTOR").empty();
    if (blocks || is_joined_line(workCtx, type_key)) {
      for (const auto& jd : blocks ? block_line_descriptors(workCtx)
                                   : joined_descriptors(workCtx, type_key)) {
        std::unique_ptr<models::D_Base> obj(
            tadah::core::factory<models::D_Base>(jd.label));
        auto m = obj->meta();
        if (!m.has_value()) {
          tadah::core::Logger::instance().log(
              tadah::core::LogLevel::INFO,
              "analysis suggest_bounds: '" + jd.label +
                  "' has no meta() yet; skipping.");
          continue;
        }
        // The rows name the descriptor as the configuration does (the
        // label of a class's metadata can be the class's: DM_Poly).
        m->label = jd.label;
        // Its ranges come from the neighbour distances of the element
        // pairs its own line names.
        auto sug = suggest_bounds_for(
            data, *m, jd.context,
            models::LinePairs(obj->get_init_atoms(jd.context)));
        sug.erase(std::remove_if(sug.begin(), sug.end(),
                                 [&](const OptimSuggestion& s) {
                                   return jd.first_entries.count(s.key) &&
                                          !jd.context.contains(s.key);
                                 }),
                  sug.end());
        for (auto& s : sug) {
          const auto it = jd.first_entries.find(s.key);
          if (it == jd.first_entries.end()) continue;
          const int before = static_cast<int>(it->second);
          s.arg_pos = s.arg_pos < 0 ? before : s.arg_pos + before;
        }
        all_suggestions.insert(all_suggestions.end(), sug.begin(), sug.end());
      }
      return;
    }
    const std::size_t n = workCtx.size(type_key);
    std::vector<std::string> labels_seen;
    for (std::size_t i = 0; i < n; ++i) {
      std::string label;
      try { label = workCtx.get<std::string>(type_key, i); }
      catch (...) { continue; }
      if (!looks_like_descriptor(label)) continue;
      // A label once.
      bool seen = false;
      for (const auto& l : labels_seen)
        if (l == label) { seen = true; break; }
      if (seen) continue;
      labels_seen.push_back(label);

      std::unique_ptr<models::D_Base> obj;
      try {
        obj.reset(tadah::core::factory<models::D_Base>(label));
      } catch (const std::exception& e) {
        tadah::core::Logger::instance().log(
            tadah::core::LogLevel::WARNING,
            "analysis suggest_bounds: cannot instantiate '" + label + "' (" +
                e.what() + "); skipping.");
        continue;
      }
      auto m = obj->meta();
      if (!m.has_value()) {
        tadah::core::Logger::instance().log(
            tadah::core::LogLevel::INFO,
            "analysis suggest_bounds: '" + label +
                "' has no meta() yet; skipping.");
        continue;
      }
      m->label = label;
      auto sug = suggest_bounds_for(
          data, *m, workCtx, models::LinePairs(obj->get_init_atoms(workCtx)));
      all_suggestions.insert(all_suggestions.end(),
                             sug.begin(), sug.end());
    }
  };

  run_for("TYPE2B");
  run_for("TYPEMB");
  if(!workCtx.blocks("DESCRIPTOR").empty())
    for(const auto &child:models::embed_contexts(workCtx))if(!child.blocks("DESCRIPTOR").empty()) {
      std::unique_ptr<models::DM_Base> descriptor(core::factory<models::DM_Base,const core::Context&>(child.get<std::string>("TYPEMB"),child));
      if(auto meta=descriptor->meta()) {
        auto suggestions=suggest_bounds_for(
            data,*meta,workCtx,
            models::LinePairs(descriptor->get_init_atoms(child)));
        all_suggestions.insert(all_suggestions.end(),suggestions.begin(),suggestions.end());
      }
    }

  // The cutoff of a D2_Knot5 or of a knot DM_REAM density, alone on its
  // TYPE line or a descriptor of a joined one, is its largest knot, which
  // tadah hpo derives: no OPTIM line.
  leave_out_derived_cutoffs(all_suggestions, workCtx);

  // One row per key and entry: the descriptors of a joined line give a
  // row each for a key they share (RCUTENV), and tadah hpo takes one OPTIM
  // line per entry.
  merge_shared_rows(all_suggestions);

  // Always print the human table to stdout for visibility.
  print_suggestions(all_suggestions, std::cout);

  // Write the OPTIM block to the chosen sink.
  write_optim_block(all_suggestions, *os_block);

  if (taskCtx.contains("OUTFILE")) {
    tadah::core::Logger::instance().log(
        tadah::core::LogLevel::INFO,
        "analysis suggest_bounds: wrote OPTIM block to " +
            taskCtx.get<std::string>("OUTFILE"));
  }

  return EXIT_SUCCESS;
}

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
