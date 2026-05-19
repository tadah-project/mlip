// Catch2 v2 unit tests for the bound_suggester (Phase 4).
//
// Verifies:
//   * suggest_bounds_for() produces one OptimSuggestion per ParamMeta and
//     iterates vector entries when the seed contains them
//   * scalar params (arg_pos == -1) emit one suggestion
//   * vector params emit one suggestion per seed entry, with arg_pos
//     advancing 0..n-1
//   * lo, hi are populated from the BoundSuggester closure
//   * write_optim_block collapses vector entries to a single (1-N) line
//   * write_optim_block emits scalar entries as (1)

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/bound_suggester.h>
#include <tadah/models/descriptors/param_meta.h>

#include <sstream>
#include <string>

using namespace tadah::mlip;
using tadah::core::Context;
using tadah::core::stats::DataStats;
using tadah::core::stats::PerPairStats;
using tadah::models::DescriptorMeta;
using tadah::models::ParamMeta;
using tadah::models::SeedView;
using tadah::models::Scale;
using tadah::models::Criticality;
using tadah::models::PairScope;
using tadah::models::HardLimit;
using tadah::models::HealthVerdict;

static DataStats make_stub_data() {
  DataStats d;
  d.n_atoms = 10;
  d.n_structures = 1;
  d.elements = {"Ta"};
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.n_pairs = 20;
  pp.r_min = 2.5;
  pp.r_max = 6.0;
  for (std::size_t i = 0; i < pp.r_percentiles.size(); ++i) {
    pp.r_percentiles[i] = 2.5 + 0.4 * static_cast<double>(i);
  }
  d.per_pair.push_back(pp);
  d.class_label = "metal_like";
  return d;
}

static DescriptorMeta make_test_meta() {
  DescriptorMeta dm;
  dm.label = "TEST";

  // Use real OPTIM keys; Context::add() rejects arbitrary names.
  ParamMeta scalar;
  scalar.key      = "RCUT2B";
  scalar.arg_pos  = -1;
  scalar.scale    = Scale::Distance;
  scalar.units    = "Å";
  scalar.role     = "scalar test param";
  scalar.crit     = Criticality::Critical;
  scalar.scope    = PairScope::Global;
  scalar.suggest  = [](const DataStats& d, const SeedView&, int) {
    auto& pp = d.per_pair[0];
    return std::make_pair(pp.r_min, pp.r_max);
  };
  dm.params.push_back(scalar);

  ParamMeta vec;
  vec.key      = "CGRID2B";
  vec.arg_pos  = 0;
  vec.scale    = Scale::Distance;
  vec.units    = "Å";
  vec.role     = "vector test param";
  vec.crit     = Criticality::Recommended;
  vec.scope    = PairScope::Global;
  vec.suggest  = [](const DataStats& d, const SeedView&, int /*ap*/) {
    auto& pp = d.per_pair[0];
    return std::make_pair(0.0, pp.r_max);
  };
  dm.params.push_back(vec);

  return dm;
}

TEST_CASE("bound_suggester: scalar param emits one suggestion", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  seed.add<double>("CGRID2B", 1.0);
  seed.add<double>("CGRID2B", 2.0);
  seed.add<double>("CGRID2B", 3.0);
  auto out = suggest_bounds_for(data, meta, seed);
  // 1 scalar + 3 vector entries
  REQUIRE(out.size() == 4);
  REQUIRE(out[0].key == "RCUT2B");
  REQUIRE(out[0].arg_pos == -1);
  REQUIRE(out[0].lo == Approx(2.5));
  REQUIRE(out[0].hi == Approx(6.0));
  REQUIRE(out[0].descriptor_label == "TEST");
}

TEST_CASE("bound_suggester: vector param iterates seed entries", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  for (int i = 0; i < 5; ++i) seed.add<double>("CGRID2B", static_cast<double>(i));
  auto out = suggest_bounds_for(data, meta, seed);
  // 1 scalar + 5 vector entries
  REQUIRE(out.size() == 6);
  for (int i = 0; i < 5; ++i) {
    auto& s = out[1 + static_cast<std::size_t>(i)];
    REQUIRE(s.key == "CGRID2B");
    REQUIRE(s.arg_pos == i);
    REQUIRE(s.lo == Approx(0.0));
    REQUIRE(s.hi == Approx(6.0));
  }
}

TEST_CASE("bound_suggester: vector param with empty seed emits one entry", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  // No CGRID2B entry — suggester should emit a single representative bound.
  auto out = suggest_bounds_for(data, meta, seed);
  REQUIRE(out.size() == 2);
  REQUIRE(out[1].key == "CGRID2B");
  REQUIRE(out[1].arg_pos == 0);
}

TEST_CASE("write_optim_block: scalar -> (1) ; vector -> (1-N)", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  for (int i = 0; i < 4; ++i) seed.add<double>("CGRID2B", static_cast<double>(i));
  auto out = suggest_bounds_for(data, meta, seed);
  std::ostringstream os;
  write_optim_block(out, os);
  const std::string txt = os.str();
  // Scalar RCUT2B should be "(1)"
  REQUIRE(txt.find("OPTIM RCUT2B (1) ") != std::string::npos);
  // Vector CGRID2B with 4 entries should collapse to "(1-4)"
  REQUIRE(txt.find("OPTIM CGRID2B (1-4) ") != std::string::npos);
}

TEST_CASE("print_suggestions: writes a header and one row per suggestion", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  seed.add<double>("CGRID2B", 0.0);
  seed.add<double>("CGRID2B", 1.0);
  auto out = suggest_bounds_for(data, meta, seed);
  std::ostringstream os;
  print_suggestions(out, os);
  const std::string txt = os.str();
  REQUIRE(txt.find("# Suggested OPTIM bounds") != std::string::npos);
  REQUIRE(txt.find("TEST.RCUT2B") != std::string::npos);
  REQUIRE(txt.find("TEST.CGRID2B[0]") != std::string::npos);
  REQUIRE(txt.find("TEST.CGRID2B[1]") != std::string::npos);
  REQUIRE(txt.find("CRITICAL") != std::string::npos);
  REQUIRE(txt.find("recommended") != std::string::npos);
}
