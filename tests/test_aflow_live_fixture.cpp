/* AFLOW: the entry the service serves today, saved on 2026-10-06 with MK's
 * approval (tests_data/aflow/):
 *   - Ta1_ICSD_76152_aflux.json, the answer of
 *     https://aflow.org/API/aflux/?auid('aflow:06436b96b885f76d');
 *   - Ta1_ICSD_76152_aflowlib.json, the entry file
 *     http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152/aflowlib.json.
 * The entry URL the provider built before (<aurl>/?format=json) answered
 * 404 "format=json not available for this entry". */
#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/data_providers/aflow_data_provider.h>
#include <tadah/mlip/structure_readers/aflow_reader.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

using tadah::core::PeriodicTable;
using tadah::mlip::AFLOWDataProvider;
using tadah::mlip::AflowReader;

namespace {

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::ostringstream os;
  os << in.rdbuf();
  return os.str();
}

/* Whether P has the static entry_url_from_aflux(json): detected, so that the
 * test compiles before the fix. */
template <class P, class = void>
struct has_entry_url : std::false_type {};
template <class P>
struct has_entry_url<P, std::void_t<decltype(P::entry_url_from_aflux(std::declval<std::string>()))>>
    : std::true_type {};

template <class P>
std::string entry_url(const std::string &aflux) {
  if constexpr (has_entry_url<P>::value)
    return P::entry_url_from_aflux(aflux);
  else
    return "";
}

}  // namespace

/* Pins: the entry AFLOW serves today, with typed fields (geometry a 6-array,
 * species and composition arrays, positions_fractional an array of
 * 3-arrays, density and volume_cell numbers), is read: one Ta atom at the
 * origin of the primitive bcc cell a = b = c = 2.90556 A, alpha = beta =
 * gamma = 109.4712 degrees (cell volume 18.8829 A^3 as the entry states),
 * labelled with its auid and compound.
 * Before the fix: nlohmann::json type_error 302 ('type must be string, but
 * is array') at the first typed field. */
TEST_CASE("AFLOW reader: the live Ta1_ICSD_76152 entry is read",
          "[aflow][structure_readers]") {
  PeriodicTable::initialize();
  AflowReader reader;
  REQUIRE_NOTHROW(reader.parseAflowJson(slurp("./tests_data/aflow/Ta1_ICSD_76152_aflowlib.json")));
  const auto &st = reader.getStructure();
  REQUIRE(st.natoms() == 1);
  CHECK(st.atoms[0].symbol == std::string("Ta"));
  for (int k = 0; k < 3; ++k) CHECK(st.atoms[0].position(k) == Approx(0.0).margin(1e-12));
  for (int i = 0; i < 3; ++i) {
    const double len = std::sqrt(st.cell(i, 0) * st.cell(i, 0) + st.cell(i, 1) * st.cell(i, 1) +
                                 st.cell(i, 2) * st.cell(i, 2));
    CHECK(len == Approx(2.90556).epsilon(1e-9));
  }
  const double vol = std::fabs(st.cell(0, 0) * (st.cell(1, 1) * st.cell(2, 2) - st.cell(1, 2) * st.cell(2, 1)) -
                               st.cell(0, 1) * (st.cell(1, 0) * st.cell(2, 2) - st.cell(1, 2) * st.cell(2, 0)) +
                               st.cell(0, 2) * (st.cell(1, 0) * st.cell(2, 1) - st.cell(1, 1) * st.cell(2, 0)));
  CHECK(vol == Approx(18.8829).epsilon(1e-4));
  CHECK_THAT(st.label, Catch::Contains("aflow:06436b96b885f76d"));
  CHECK_THAT(st.label, Catch::Contains("Ta1"));
}

/* Pins: the provider's second request is the entry file <aurl>/aflowlib.json:
 * from the AFLUX answer the service gives today,
 * AFLOWDataProvider::entry_url_from_aflux() builds
 * http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152/aflowlib.json.
 * Before the fix: the provider requested the entry's directory with
 * "?format=json" (aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152?format=json,
 * built inline in fetchAndParseAflowByAuid), which the service answers with
 * 404; with no entry_url_from_aflux, has_entry_url is false and entry_url()
 * gives "". */
TEST_CASE("AFLOW provider: the entry URL is the entry's aflowlib.json",
          "[aflow][data_providers]") {
  CHECK(has_entry_url<AFLOWDataProvider>::value);
  CHECK(entry_url<AFLOWDataProvider>(slurp("./tests_data/aflow/Ta1_ICSD_76152_aflux.json")) ==
        "http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152/aflowlib.json");
}
