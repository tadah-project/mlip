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

#include <limits>
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
    /* core::parse_fixindex's message, the one every site gives (before
     * the fix: "FIXINDEX entry 6 is out of range: this model has 5 weights
     * (valid range 1..5).", which this section checked for "out of
     * range"). */
    Context c = ctx_with(true, "6", {0.5});
    REQUIRE_THROWS_WITH(
        classify_weight_source(c, kNCols),
        "FIXINDEX: index 6 is outside 1-5 (the model has 5 weights)");
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

/* Pins: classify_weight_source reads FIXINDEX with core::parse_fixindex:
 *  - FIXINDEX given as separate values pairs like the comma list: '2 4'
 *    with two FIXWEIGHT values is PartialLS and '1 2 3 4 5' without a
 *    DBFILE FixedOnlyNoData, as '2,4' and '1-5' are;
 *  - '2-1' is refused as a descending range, '2,2' as a weight named twice
 *    (two FIXWEIGHT values) or as 2 indices for 1 FIXWEIGHT value, and '0'
 *    as outside 1-5 (the model has 5 weights).
 * Before the fix: the lenient core::parse_indices dropped '2 4' and
 * '1 2 3 4 5' (the values joined into one malformed segment) and expanded
 * '2-1' to nothing: "FIXINDEX names 0 weights but FIXWEIGHT provides 2
 * values"; '2,2' was deduplicated to weight 2, so with one FIXWEIGHT value
 * it was PartialLS and with two "FIXINDEX names 1 weights but FIXWEIGHT
 * provides 2 values"; '0' was "FIXINDEX entry 0 is out of range". */
TEST_CASE("classify_weight_source: FIXINDEX is read strictly, separate values "
          "as one list",
          "[weight_provisioning][fixindex]") {
  SECTION("separate values pair like the comma list") {
    Context partial = ctx_with(true, "2 4", {0.1, 0.2});
    REQUIRE(partial.size("FIXINDEX") == 2);
    CHECK(classify_weight_source(partial, kNCols) == WeightSource::PartialLS);
    Context all = ctx_with(false, "1 2 3 4 5", {1, 2, 3, 4, 5});
    CHECK(classify_weight_source(all, kNCols) ==
          WeightSource::FixedOnlyNoData);
  }
  SECTION("a descending range, a repeat and index 0 are refused") {
    Context desc = ctx_with(true, "2-1", {0.1, 0.2});
    CHECK_THROWS_WITH(classify_weight_source(desc, kNCols),
                      Catch::Contains("FIXINDEX: index '2-1' is a descending "
                                      "range"));
    Context twice = ctx_with(true, "2,2", {0.1, 0.2});
    CHECK_THROWS_WITH(classify_weight_source(twice, kNCols),
                      Catch::Contains("FIXINDEX 2,2 names weight 2 twice"));
    Context one = ctx_with(true, "2,2", {0.1});
    CHECK_THROWS_WITH(classify_weight_source(one, kNCols),
                      Catch::Contains("FIXINDEX 2,2 has 2 indices but "
                                      "FIXWEIGHT has 1 value"));
    Context zero = ctx_with(true, "0", {0.1});
    CHECK_THROWS_WITH(
        classify_weight_source(zero, kNCols),
        "FIXINDEX: index 0 is outside 1-5 (the model has 5 weights)");
  }
}

/* Pins: provision_fixed_weights takes FIXINDEX given as separate values
 * ('K 1 2 ... K-1', one value each) as the comma list: FIXWEIGHT[0] pins
 * weight K, FIXWEIGHT[i] weight i.
 * Before the fix: "FIXINDEX names 0 weights but FIXWEIGHT provides K
 * values" (the lenient parser dropped the joined values). */
TEST_CASE("provision_fixed_weights: FIXINDEX given as separate values",
          "[weight_provisioning][fixindex]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  std::size_t K = 0;
  {
    Context probe_ctx = make_model_context();
    Trainer probe(probe_ctx, ws);
    K = probe.fb->get_phi_cols(probe.context);
  }
  REQUIRE(K >= 2);
  Context ctx = make_model_context();
  {
    std::ostringstream fi;  // K 1 2 ... K-1
    fi << K;
    for (std::size_t i = 1; i < K; ++i) fi << " " << i;
    ctx.add("FIXINDEX", fi.str());
  }
  REQUIRE(ctx.size("FIXINDEX") == K);
  for (std::size_t k = 0; k < K; ++k)
    ctx.add("FIXWEIGHT", 0.5 + static_cast<double>(k));
  Trainer tr(ctx, ws);
  REQUIRE_NOTHROW(tr.provision_fixed_weights());
  const auto &w = tr.model->get_weights();
  REQUIRE(w.size() == K);
  CHECK(w[K - 1] == 0.5);
  for (std::size_t i = 1; i < K; ++i)
    CHECK(w[i - 1] == 0.5 + static_cast<double>(i));
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

TEST_CASE("provision_fixed_weights: a subnormal FIXWEIGHT is a number",
          "[weight_provisioning][lossless]") {
  // Context::add(double) now keeps a subnormal ("1e-310"; the old
  // fixed-decimal text flushed it to 0), and the refit writer / HPO put
  // computed values into FIXWEIGHT. std::stod rejected it (ERANGE):
  // "FIXWEIGHT entry is not a number".
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  std::size_t K = 0;
  {
    Context probe_ctx = make_model_context();
    Trainer probe(probe_ctx, ws);
    K = probe.fb->get_phi_cols(probe.context);
  }
  REQUIRE(K >= 2);
  Context ctx = make_model_context();
  ctx.add("FIXINDEX", "1-" + std::to_string(K));
  ctx.add("FIXWEIGHT", 1e-310);
  for (std::size_t k = 1; k < K; ++k)
    ctx.add("FIXWEIGHT", -std::numeric_limits<double>::denorm_min());
  Trainer tr(ctx, ws);
  REQUIRE_NOTHROW(tr.provision_fixed_weights());
  const auto &w = tr.model->get_weights();
  REQUIRE(w.size() == K);
  CHECK(w[0] == 1e-310);
  CHECK(w[K - 1] == -std::numeric_limits<double>::denorm_min());
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

/* Pins: a Trainer whose configuration carries a SIGMA (that of a potential
 * used as the configuration, here N = K and K*K values) holds none of it
 * once built (Trainer::training_context: neither its context nor the
 * potential it writes, get_param_file, has the key), and after it pins
 * every weight with FIXINDEX/FIXWEIGHT (provision_fixed_weights, no fit)
 * still writes no SIGMA, with the FIXWEIGHT values for weights.
 * provision_fixed_weights' own removal, of a SIGMA set in the context after
 * the Trainer was built, is pinned in test_trainer_sigma.cpp
 * ("provision_fixed_weights removes the SIGMA of an earlier fit of its
 * Trainer").
 * Before the fix (neither Trainer::training_context nor
 * provision_fixed_weights removed a SIGMA): both kept the configuration's
 * SIGMA, so the potential (an HPO best_pot.tadah with every weight pinned)
 * paired a covariance of other weights with the pinned ones. */
TEST_CASE("provision_fixed_weights writes no SIGMA of the configuration",
          "[weight_provisioning][sigma]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  std::size_t K = 0;
  {
    Context probe_ctx = make_model_context();
    Trainer probe(probe_ctx, ws);
    K = probe.fb->get_phi_cols(probe.context);
  }
  REQUIRE(K >= 2);
  Context ctx = make_model_context();
  ctx.add("FIXINDEX", "1-" + std::to_string(K));
  for (std::size_t k = 0; k < K; ++k)
    ctx.add("FIXWEIGHT", 0.5 + static_cast<double>(k));
  ctx.add("SIGMA", K);
  for (std::size_t k = 0; k < K * K; ++k) ctx.add("SIGMA", 0.125);

  Trainer tr(ctx, ws);
  CHECK_FALSE(tr.context.has_stored("SIGMA"));
  CHECK_FALSE(tr.get_param_file().has_stored("SIGMA"));
  REQUIRE_NOTHROW(tr.provision_fixed_weights());
  CHECK_FALSE(tr.context.has_stored("SIGMA"));
  CHECK_FALSE(tr.get_param_file().has_stored("SIGMA"));
  const auto &w = tr.model->get_weights();
  REQUIRE(w.size() == K);
  for (std::size_t k = 0; k < K; ++k)
    CHECK(w[k] == 0.5 + static_cast<double>(k));
}
