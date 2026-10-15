// test_weight_covariance_io.cpp
//
// The format of a potential's weight covariance (weight_covariance.h):
// write_weight_covariance writes WCOV (N, then N x N values column-major)
// and WNOISE (row type and noise variance pairs, E F S order, the set ones
// only), replacing SIGMA, WCOV and WNOISE; read_weight_covariance reads
// them back and refuses what does not fit the potential before reading a
// value. These replace LinearRegressor::config_add_sigma and read_sigma
// (MODELS), removed with SIGMA: the count checks of read_sigma (N = 2
// with 4 values, N = 4 with 34) are pinned here for WCOV.
//
// New API: there is no before (the end-to-end failures before the fix are
// those of test_predict_error.cpp).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/mlip/normaliser.h>
#include <tadah/mlip/weight_covariance.h>

#include <string>

using tadah::core::Context;
using tadah::core::Matrix;
using namespace tadah::mlip;

namespace {

std::string refusal(const Context &c, std::size_t n) {
  try {
    read_weight_covariance(c, n);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "<no exception>";
}

}  // namespace

/* Pins: a 2 x 2 covariance and the noise of E and S written and read back
 * bit for bit (WCOV "2 a b c d" column-major, WNOISE "E e S s", no F);
 * writing replaces a SIGMA, WCOV and WNOISE the Context held. */
TEST_CASE("WCOV and WNOISE: write and read back", "[weight_covariance]") {
  Context c;
  c.add("SIGMA", "2 1 0 0 1");
  c.add("WCOV", "3 1 0 0 0 1 0 0 0 1");
  c.add("WNOISE", "F 9");
  Matrix S(2, 2);
  S(0, 0) = 0.5;
  S(1, 0) = 0.125;
  S(0, 1) = 0.125;
  S(1, 1) = 0.25;
  PerRowType noise;
  noise.value[kEnergyRow] = 0.04;
  noise.set[kEnergyRow] = true;
  noise.value[kStressRow] = 1e-6;
  noise.set[kStressRow] = true;
  write_weight_covariance(c, S, noise);
  CHECK_FALSE(c.has_stored("SIGMA"));
  REQUIRE(c.size("WCOV") == 5);
  CHECK(c.get<std::size_t>("WCOV") == 2);
  CHECK(c.get<double>("WCOV", 2) == 0.125);  // (1, 0)
  REQUIRE(c.size("WNOISE") == 4);
  CHECK(c.get<std::string>("WNOISE", 0) == "E");
  CHECK(c.get<std::string>("WNOISE", 2) == "S");
  const WeightCovariance wc = read_weight_covariance(c, 2);
  REQUIRE(wc.cov.rows() == 2);
  for (std::size_t a = 0; a < 2; ++a)
    for (std::size_t b = 0; b < 2; ++b) CHECK(wc.cov(a, b) == S(a, b));
  CHECK(wc.noise.set[kEnergyRow]);
  CHECK_FALSE(wc.noise.set[kForceRow]);
  CHECK(wc.noise.set[kStressRow]);
  CHECK(wc.noise.value[kEnergyRow] == 0.04);
  CHECK(wc.noise.value[kStressRow] == 1e-6);
}

/* Pins: read_weight_covariance refuses, naming WCOV or WNOISE: no WCOV or
 * no WNOISE (naming OALGO 3; why an OALGO 3 potential has none is the case
 * below); N = 2 for 3 weights ("N = 2", "3 WEIGHTS"); N = 2 with 3 values
 * after it ("5 values", "has 4") and N = 4 with 33 ("17 values", "has 34",
 * the count of two appended SIGMAs before fix/hpo-trainer-reuse); a WNOISE
 * of odd length, of an unknown type, with a type twice, or with a negative
 * variance. (A WNOISE without E, "F 0.5", was refused here too until WCOV
 * and WNOISE were revised: it is the WNOISE of a fit without energies,
 * read by the case below. This Context without WCOV, with no OALGO and so
 * the default OALGO 1, was expected to name SIGMA too, which only an OALGO
 * 3 potential had.) */
TEST_CASE("read_weight_covariance refuses what does not fit the potential",
          "[weight_covariance]") {
  Context ok;
  ok.add("WCOV", "2 1 0 0 1");
  ok.add("WNOISE", "E 0.5");
  REQUIRE(refusal(ok, 2) == "<no exception>");
  {
    Context c;
    c.add("WNOISE", "E 0.5");
    const std::string m = refusal(c, 2);
    INFO(m);
    CHECK(m.find("WCOV") != std::string::npos);
    CHECK(m.find("OALGO 3") != std::string::npos);
  }
  {
    Context c;
    c.add("WCOV", "2 1 0 0 1");
    CHECK(refusal(c, 2).find("WNOISE") != std::string::npos);
  }
  {
    const std::string m = refusal(ok, 3);
    INFO(m);
    CHECK(m.find("N = 2") != std::string::npos);
    CHECK(m.find("3 WEIGHTS") != std::string::npos);
  }
  {
    Context c = ok;
    c.remove("WCOV");
    c.add("WCOV", "2 1 0 0");
    const std::string m = refusal(c, 2);
    INFO(m);
    CHECK(m.find("5 values") != std::string::npos);
    CHECK(m.find("has 4") != std::string::npos);
  }
  {
    Context c = ok;
    c.remove("WCOV");
    c.add("WCOV", "4");
    for (int k = 0; k < 33; ++k) c.add("WCOV", 0.5);
    const std::string m = refusal(c, 4);
    INFO(m);
    CHECK(m.find("N = 4") != std::string::npos);
    CHECK(m.find("17 values") != std::string::npos);
    CHECK(m.find("has 34") != std::string::npos);
  }
  for (const char *noise : {"E 0.5 F", "X 0.5", "E 0.5 E 0.25", "E -0.5"}) {
    INFO("WNOISE " << noise);
    Context c = ok;
    c.remove("WNOISE");
    for (const char *t : {noise}) {
      std::string s(t);
      std::size_t at = 0;
      while (at < s.size()) {
        const std::size_t sp = s.find(' ', at);
        c.add("WNOISE", s.substr(at, sp == std::string::npos ? std::string::npos
                                                              : sp - at));
        if (sp == std::string::npos) break;
        at = sp + 1;
      }
    }
    const std::string m = refusal(c, 2);
    INFO(m);
    CHECK(m.find("WNOISE") != std::string::npos);
  }
}

/* Pins: read_weight_covariance reads the WNOISE of a fit without energies
 * (EWEIGHT 0, or every structure's eweight 0), which lists F (and S) and no
 * E, as write_weight_covariance writes it: "F 0.5" read with F set to 0.5
 * and E and S unset.
 * Before the fix: refused, "WNOISE lists no E (energies): every prediction
 * has an energy. `tadah train` writes WCOV and WNOISE for a fit with OALGO
 * 3; retrain the potential with OALGO 3." (the writer wrote what its reader
 * refused, and the remedy was that of a potential without OALGO 3). */
TEST_CASE("read_weight_covariance reads the WNOISE of a fit without energies",
          "[weight_covariance]") {
  Context c;
  c.add("OALGO", 3);
  c.add("WCOV", "2 1 0 0 1");
  c.add("WNOISE", "F");
  c.add("WNOISE", 0.5);
  REQUIRE(refusal(c, 2) == "<no exception>");
  const WeightCovariance wc = read_weight_covariance(c, 2);
  CHECK_FALSE(wc.noise.set[kEnergyRow]);
  CHECK(wc.noise.set[kForceRow]);
  CHECK_FALSE(wc.noise.set[kStressRow]);
  CHECK(wc.noise.value[kForceRow] == 0.5);
}

/* Pins: a potential without WCOV and WNOISE is refused with the reason its
 * configuration gives: with OALGO 1 the message names OALGO 1 and says to
 * retrain with OALGO 3; with OALGO 3 (M_BLR BF_Linear) it names the three
 * ways such a potential has none ("SIGMA": a potential of an earlier
 * version; "observations": a fit with no more observations than effective
 * weights; "FIXWEIGHT": weights set without a fit) and does not say to
 * retrain with OALGO 3; with OALGO 3 and MODEL M_KRR Kern_RBF 2 it names
 * implementation 2 and the remedy "MODEL M_KRR <kernel> 1".
 * Before the fix: every one said "`tadah train` writes WCOV and WNOISE for
 * a fit with OALGO 3; retrain the potential with OALGO 3" (the OALGO 3
 * potentials already had it, and retraining them alike gave the same
 * potential), naming neither the observations, FIXWEIGHT nor
 * implementation 2. */
TEST_CASE("read_weight_covariance says why a potential has no covariance",
          "[weight_covariance]") {
  {
    Context c;
    c.add("OALGO", 1);
    c.add("MODEL", "M_BLR BF_Linear");
    const std::string m = refusal(c, 2);
    INFO(m);
    CHECK(m.find("WCOV") != std::string::npos);
    CHECK(m.find("OALGO 1") != std::string::npos);
    CHECK(m.find("retrain the potential with OALGO 3") != std::string::npos);
  }
  {
    Context c;
    c.add("OALGO", 3);
    c.add("MODEL", "M_BLR BF_Linear");
    const std::string m = refusal(c, 2);
    INFO(m);
    CHECK(m.find("WCOV") != std::string::npos);
    CHECK(m.find("SIGMA") != std::string::npos);
    CHECK(m.find("observations") != std::string::npos);
    CHECK(m.find("FIXWEIGHT") != std::string::npos);
    CHECK(m.find("retrain the potential with OALGO 3") == std::string::npos);
  }
  {
    Context c;
    c.add("OALGO", 3);
    c.add("MODEL", "M_KRR Kern_RBF 2");
    const std::string m = refusal(c, 2);
    INFO(m);
    CHECK(m.find("WCOV") != std::string::npos);
    CHECK(m.find("implementation 2") != std::string::npos);
    CHECK(m.find("MODEL M_KRR <kernel> 1") != std::string::npos);
    CHECK(m.find("retrain the potential with OALGO 3") == std::string::npos);
  }
}

/* Pins: the refusal of an OALGO 3 potential (M_BLR BF_Linear) without WCOV
 * and WNOISE says, of the potentials with only SIGMA, exactly "a potential
 * of an earlier version of Tadah!, whose SIGMA is ignored (it was computed
 * with a transposed formula): retrain it;" (an OALGO 3 potential of an
 * earlier version has SIGMA), and names no development stage
 * ("development").
 * Before the fix: it said "a potential of Tadah! 1.0.0 to 1.5.0 or a 1.6.0
 * development build, whose SIGMA is ignored ...": both checks fail. */
TEST_CASE("read_weight_covariance: an OALGO 3 potential with SIGMA only is "
          "one of an earlier version",
          "[weight_covariance]") {
  Context c;
  c.add("OALGO", 3);
  c.add("MODEL", "M_BLR BF_Linear");
  const std::string m = refusal(c, 2);
  INFO(m);
  CHECK(m.find("a potential of an earlier version of Tadah!, whose SIGMA is "
               "ignored (it was computed with a transposed formula): retrain "
               "it;") != std::string::npos);
  CHECK(m.find("development") == std::string::npos);
}

/* Pins: require_noise_of refuses every prediction of a covariance whose
 * WNOISE has no E (the fit had no energies with a non-zero weight): every
 * prediction has an energy, so energies alone and forces with F are both
 * refused, naming the energies, WNOISE and EWEIGHT 0 (the cause), not
 * OALGO 3.
 * Before the fix: require_noise_of checked only forces and stresses, so
 * both passed (read_weight_covariance refused such a WNOISE first, with the
 * remedy of a potential without OALGO 3). */
TEST_CASE("require_noise_of refuses the energies of a fit without them",
          "[weight_covariance]") {
  WeightCovariance wc;
  wc.noise.set[kForceRow] = true;
  for (const bool force : {false, true}) {
    INFO("force " << force);
    std::string m = "<no exception>";
    try {
      require_noise_of(wc, force, false);
    } catch (const std::exception &e) {
      m = e.what();
    }
    INFO(m);
    CHECK(m.find("energies") != std::string::npos);
    CHECK(m.find("WNOISE") != std::string::npos);
    CHECK(m.find("EWEIGHT 0") != std::string::npos);
    CHECK(m.find("OALGO 3") == std::string::npos);
  }
}

/* Pins: require_noise_of refuses forces of a covariance whose WNOISE has
 * no F, and stresses without S, naming the quantity and WNOISE; it accepts
 * energies alone, and forces with F. */
TEST_CASE("require_noise_of refuses a quantity the fit had no noise of",
          "[weight_covariance]") {
  WeightCovariance wc;
  wc.noise.set[kEnergyRow] = true;
  CHECK_NOTHROW(require_noise_of(wc, false, false));
  CHECK_THROWS_WITH(require_noise_of(wc, true, false),
                    Catch::Contains("forces") && Catch::Contains("WNOISE"));
  CHECK_THROWS_WITH(require_noise_of(wc, false, true),
                    Catch::Contains("stresses") && Catch::Contains("WNOISE"));
  wc.noise.set[kForceRow] = true;
  CHECK_NOTHROW(require_noise_of(wc, true, false));
}

/* Pins: raw_descriptor_map for 2 intercept columns and 2 descriptor columns
 * (means 1, 2 and stdevs 2, 4 of columns 2, 3): J(i,i) = 1/stdev_i, the
 * intercept rows 1, 0, -1/2, -1/2 and 0, 1, -1/2, -1/2 (the same
 * correction on every species' intercept), and with no intercept column it
 * is refused naming BIAS. */
TEST_CASE("raw_descriptor_map maps normalised weights to raw ones",
          "[weight_covariance]") {
  Normaliser norm;
  norm.mean = tadah::core::v_type(4);
  norm.std_dev = tadah::core::v_type(4);
  norm.mean[0] = norm.mean[1] = 0.0;
  norm.std_dev[0] = norm.std_dev[1] = 0.0;
  norm.mean[2] = 1.0;
  norm.mean[3] = 2.0;
  norm.std_dev[2] = 2.0;
  norm.std_dev[3] = 4.0;
  const Matrix J = raw_descriptor_map(norm, 4, 2);
  const double expect[4][4] = {{1, 0, -0.5, -0.5},
                               {0, 1, -0.5, -0.5},
                               {0, 0, 0.5, 0},
                               {0, 0, 0, 0.25}};
  for (std::size_t a = 0; a < 4; ++a)
    for (std::size_t b = 0; b < 4; ++b) {
      INFO(a << ", " << b);
      CHECK(J(a, b) == expect[a][b]);
    }
  CHECK_THROWS_WITH(raw_descriptor_map(norm, 4, 0), Catch::Contains("BIAS"));
}
