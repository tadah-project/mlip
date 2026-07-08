// test_weight_provisioning.cpp
//
// WeightSource classification and direct weight provisioning from
// FIXINDEX/FIXWEIGHT (the no-training-data path used by `tadah hpo` when no
// DBFILE is given, and by the all-fixed coefficient-search workload).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/weight_provisioning.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Trainer;
using tadah::mlip::WeightSource;
using tadah::mlip::classify_weight_source;
using tadah::mlip::provision_fixed_weights;
using tadah::mlip::memory::MLIPWorkspaceManager;

constexpr std::size_t kNCols = 5;

Context ctx_with(bool dbfile, const std::string &fixindex,
                 const std::vector<double> &fixweight) {
  Context c;
  if (dbfile) c.add("DBFILE", "dummy.tadahdb");  // classification never opens it
  if (!fixindex.empty()) c.add("FIXINDEX", fixindex);
  for (double v : fixweight) c.add("FIXWEIGHT", v);
  return c;
}

/* Minimal D2_BP + BF_Linear context (same fixture family as
 * test_m_tadah_base_w_grad.cpp) — just the keys M_BLR + Trainer +
 * DC_Selector need. */
Context make_model_context() {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "H");
  c.add("TYPE2B", "H");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  c.add("CGRID2B", "1.0 2.0");
  c.add("SGRID2B", "0.5 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("RCUTMB", 5.0);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

}  // namespace

TEST_CASE("classify_weight_source: the four modes", "[weight_provisioning]") {
  SECTION("DBFILE only -> FullLS") {
    Context c = ctx_with(true, "", {});
    REQUIRE(classify_weight_source(c, kNCols) == WeightSource::FullLS);
  }
  SECTION("DBFILE + partial pin -> PartialLS") {
    Context c = ctx_with(true, "2,4", {0.1, 0.2});
    REQUIRE(classify_weight_source(c, kNCols) == WeightSource::PartialLS);
  }
  SECTION("DBFILE + full pin -> AllFixedData") {
    Context c = ctx_with(true, "1-5", {1, 2, 3, 4, 5});
    REQUIRE(classify_weight_source(c, kNCols) == WeightSource::AllFixedData);
  }
  SECTION("no DBFILE + full pin -> FixedOnlyNoData") {
    Context c = ctx_with(false, "1-5", {1, 2, 3, 4, 5});
    REQUIRE(classify_weight_source(c, kNCols) == WeightSource::FixedOnlyNoData);
  }
}

TEST_CASE("classify_weight_source: illegal combinations", "[weight_provisioning]") {
  SECTION("lone FIXINDEX") {
    Context c = ctx_with(true, "2", {});
    REQUIRE_THROWS_WITH(classify_weight_source(c, kNCols),
                        Catch::Contains("requires"));
  }
  SECTION("lone FIXWEIGHT") {
    Context c = ctx_with(true, "", {0.5});
    REQUIRE_THROWS_WITH(classify_weight_source(c, kNCols),
                        Catch::Contains("requires"));
  }
  SECTION("size mismatch") {
    Context c = ctx_with(true, "1,2,3", {0.5});
    REQUIRE_THROWS_WITH(classify_weight_source(c, kNCols),
                        Catch::Contains("pair 1:1"));
  }
  SECTION("out-of-bounds index") {
    Context c = ctx_with(true, "6", {0.5});
    REQUIRE_THROWS_WITH(classify_weight_source(c, kNCols),
                        Catch::Contains("out of range"));
  }
  SECTION("no DBFILE, partial coverage lists uncovered weights") {
    Context c = ctx_with(false, "2", {0.5});
    REQUIRE_THROWS_WITH(classify_weight_source(c, kNCols),
                        Catch::Contains("1,3-5"));
  }
  SECTION("no DBFILE, no pinning at all") {
    Context c;
    REQUIRE_THROWS_WITH(classify_weight_source(c, kNCols),
                        Catch::Contains("cannot be determined"));
  }
}

TEST_CASE("provision_fixed_weights: scatter + trained flag on real M_BLR",
          "[weight_provisioning]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;

  // Probe the model dimension first, then pin every weight with a permuted
  // FIXINDEX so the declaration-order pairing is exercised.
  std::size_t K = 0;
  {
    Context probe_ctx = make_model_context();
    Trainer probe(probe_ctx, ws);
    K = probe.fb->get_phi_cols(probe.context);
  }
  REQUIRE(K >= 2);

  Context ctx = make_model_context();
  {
    std::ostringstream fi;  // K,1,2,...,K-1
    fi << K;
    for (std::size_t i = 1; i < K; ++i) fi << "," << i;
    ctx.add("FIXINDEX", fi.str());
  }
  for (std::size_t k = 0; k < K; ++k)
    ctx.add("FIXWEIGHT", 0.5 + static_cast<double>(k));

  Trainer tr(ctx, ws);
  REQUIRE_FALSE(tr.model->is_trained());
  tr.provision_fixed_weights();

  REQUIRE(tr.model->is_trained());
  const auto &w = tr.model->get_weights();
  REQUIRE(w.size() == K);
  // FIXINDEX = [K, 1, 2, ..., K-1]; FIXWEIGHT[k] = 0.5 + k.
  REQUIRE(w[K - 1] == 0.5);
  for (std::size_t i = 1; i < K; ++i)
    REQUIRE(w[i - 1] == 0.5 + static_cast<double>(i));
}

TEST_CASE("provision_fixed_weights: incomplete coverage throws",
          "[weight_provisioning]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;

  Context ctx = make_model_context();
  ctx.add("FIXINDEX", "1");
  ctx.add("FIXWEIGHT", 1.0);

  Trainer tr(ctx, ws);
  REQUIRE_THROWS_WITH(tr.provision_fixed_weights(),
                      Catch::Contains("not covered by FIXINDEX"));
}
