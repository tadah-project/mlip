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
 *    (with two FIXWEIGHT values or one), and '0' as outside 1-5 (the model
 *    has 5 weights).
 * Before the fix: the lenient core::parse_indices dropped '2 4' and
 * '1 2 3 4 5' (the values joined into one malformed segment) and expanded
 * '2-1' to nothing: "FIXINDEX names 0 weights but FIXWEIGHT provides 2
 * values"; '2,2' was deduplicated to weight 2, so with one FIXWEIGHT value
 * it was PartialLS and with two "FIXINDEX names 1 weights but FIXWEIGHT
 * provides 2 values"; '0' was "FIXINDEX entry 0 is out of range". After
 * core::parse_fixindex first read FIXINDEX, '2,2' with one value was
 * refused as "FIXINDEX 2,2 has 2 indices but FIXWEIGHT has 1 value" (this
 * case pinned that): it compared the count before it looked for a
 * repeat. */
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
                      Catch::Contains("FIXINDEX 2,2 names weight 2 twice"));
    Context zero = ctx_with(true, "0", {0.1});
    CHECK_THROWS_WITH(
        classify_weight_source(zero, kNCols),
        "FIXINDEX: index 0 is outside 1-5 (the model has 5 weights)");
  }
}

/* Pins: classify_weight_source (tadah hpo's startup check) refuses
 * FIXINDEX/FIXWEIGHT for M_KRR implementation 2 (MODEL M_KRR Kern_Linear 2
 * and Kern_RBF 2), whose solve cannot pin weights, before it reads
 * FIXINDEX: FIXINDEX 1 / FIXWEIGHT 14 and FIXINDEX 99 (past any weight) are
 * both "MODEL M_KRR <kernel> 2 cannot pin weights"; implementation 1 (the
 * same kernels with 1) classifies FIXINDEX 1 as PartialLS, as before.
 * Before the fix: implementation 2 was classified as implementation 1:
 * FIXINDEX 1 was PartialLS (every evaluation then trained with the pin
 * ignored) and 99 "FIXINDEX: index 99 is outside 1-5 (the model has 5
 * weights)". */
TEST_CASE("classify_weight_source: M_KRR implementation 2 cannot pin weights",
          "[weight_provisioning][fixindex][krr]") {
  for (const char *kern : {"Kern_Linear", "Kern_RBF"}) DYNAMIC_SECTION(kern) {
    auto with_model = [&](const std::string &impl, const std::string &fi) {
      Context c = ctx_with(true, fi, {14.0});
      c.add("MODEL", "M_KRR");
      c.add("MODEL", kern);
      c.add("MODEL", impl);
      return c;
    };
    const std::string refusal =
        std::string("MODEL M_KRR ") + kern + " 2 cannot pin weights";
    CHECK_THROWS_WITH(classify_weight_source(with_model("2", "1"), kNCols),
                      Catch::StartsWith(refusal));
    CHECK_THROWS_WITH(classify_weight_source(with_model("2", "99"), kNCols),
                      Catch::StartsWith(refusal));
    CHECK(classify_weight_source(with_model("1", "1"), kNCols) ==
          WeightSource::PartialLS);
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

/* Pins: provision_fixed_weights on a configuration with neither FIXINDEX
 * nor FIXWEIGHT names every weight as uncovered ("provision_fixed_weights:
 * every model weight must be pinned. Weights 1-K are not covered by
 * FIXINDEX."), not the lone-key message: neither key is lone.
 * Before the fix: it threw the lone-key message ("FIXINDEX requires
 * FIXWEIGHT (and vice versa). ...") when either key was missing, so also
 * when both were. */
TEST_CASE("provision_fixed_weights without FIXINDEX and FIXWEIGHT names the "
          "uncovered weights",
          "[weight_provisioning][fixindex]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  Context ctx = make_model_context();
  REQUIRE_FALSE(ctx.has_stored("FIXINDEX"));
  REQUIRE_FALSE(ctx.has_stored("FIXWEIGHT"));
  Trainer tr(ctx, ws);
  const std::size_t K = tr.fb->get_phi_cols(tr.context);
  REQUIRE(K >= 2);
  REQUIRE_THROWS_WITH(
      tr.provision_fixed_weights(),
      "provision_fixed_weights: every model weight must be pinned. Weights "
      "1-" + std::to_string(K) + " are not covered by FIXINDEX.");
  CHECK_FALSE(tr.model->is_trained());
}

/* Pins: a Trainer whose configuration carries the covariance of another
 * fit (SIGMA, WCOV and WNOISE, those of a potential used as the
 * configuration: N = K and K*K values) holds none of it once built
 * (Trainer::training_context: neither its context nor the potential it
 * writes, get_param_file, has the keys), and after it pins every weight
 * with FIXINDEX/FIXWEIGHT (provision_fixed_weights, no fit) still writes
 * none of them, with the FIXWEIGHT values for weights.
 * provision_fixed_weights after a fit of its Trainer is pinned in
 * test_trainer_sigma.cpp ("provision_fixed_weights leaves no covariance
 * of an earlier fit of its Trainer").
 * Before the fix (WCOV and WNOISE): WCOV and WNOISE were not keys
 * (Context::add threw "Key not allowed: WCOV"), and training_context
 * removed SIGMA only. Before fix/hpo-trainer-reuse (neither
 * Trainer::training_context nor provision_fixed_weights removed a SIGMA):
 * both kept the configuration's SIGMA, so the potential (an HPO
 * best_pot.tadah with every weight pinned) paired a covariance of other
 * weights with the pinned ones. */
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
  for (const char *key : {"SIGMA", "WCOV"}) {
    ctx.add(key, K);
    for (std::size_t k = 0; k < K * K; ++k) ctx.add(key, 0.125);
  }
  ctx.add("WNOISE", "E 0.5");

  Trainer tr(ctx, ws);
  for (const char *key : {"SIGMA", "WCOV", "WNOISE"}) {
    INFO(key);
    CHECK_FALSE(tr.context.has_stored(key));
    CHECK_FALSE(tr.get_param_file().has_stored(key));
  }
  REQUIRE_NOTHROW(tr.provision_fixed_weights());
  for (const char *key : {"SIGMA", "WCOV", "WNOISE"}) {
    INFO(key);
    CHECK_FALSE(tr.context.has_stored(key));
    CHECK_FALSE(tr.get_param_file().has_stored(key));
  }
  const auto &w = tr.model->get_weights();
  REQUIRE(w.size() == K);
  for (std::size_t k = 0; k < K; ++k)
    CHECK(w[k] == 0.5 + static_cast<double>(k));
}

/* Pins: each refusal of classify_weight_source (tadah hpo at startup) of a
 * configuration without a training DBFILE ends with the advice every
 * refusal of a configuration without one gives, "A potential records no
 * DBFILE (no potential tadah writes has one): to train from one, add a
 * DBFILE line to a copy of it (the Training page of the documentation,
 * Retraining from a potential).", after its own reason: no FIXINDEX or
 * FIXWEIGHT ("the model weights cannot be determined"; a potential used as
 * the -c configuration as it is gets this one), FIXINDEX 2 of 5 weights
 * ("every model weight must be pinned"), and M_KRR Kern_RBF ("draws its
 * SBASIS basis vectors from the training data").
 * Before the fix: each refusal ended with its own remedy ("... or pin every
 * weight with FIXINDEX/FIXWEIGHT.", "... to search them).", "Provide a
 * training DBFILE."), without the advice. */
TEST_CASE("classify_weight_source: a refusal without a training DBFILE says "
          "that a potential records none",
          "[weight_provisioning][potential_content]") {
  const std::string advice =
      "A potential records no DBFILE (no potential tadah writes has one): "
      "to train from one, add a DBFILE line to a copy of it (the Training "
      "page of the documentation, Retraining from a potential).";
  SECTION("no FIXINDEX or FIXWEIGHT") {
    CHECK_THROWS_WITH(classify_weight_source(ctx_with(false, "", {}), kNCols),
                      Catch::Contains("the model weights cannot be "
                                      "determined") &&
                          Catch::EndsWith(advice));
  }
  SECTION("pins that do not cover every weight") {
    CHECK_THROWS_WITH(
        classify_weight_source(ctx_with(false, "2", {0.5}), kNCols),
        Catch::Contains("every model weight must be pinned") &&
            Catch::EndsWith(advice));
  }
  SECTION("a basis drawn from the training data") {
    Context c = ctx_with(false, "", {});
    c.add("MODEL", "M_KRR");
    c.add("MODEL", "Kern_RBF");
    c.add("MODEL", 1);
    CHECK_THROWS_WITH(classify_weight_source(c, kNCols),
                      Catch::Contains("draws its SBASIS basis vectors from "
                                      "the training data") &&
                          Catch::EndsWith(advice));
  }
}
