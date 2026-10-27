/* AFLOWDataProvider and the AflowReader that uses it, without the network.
 *
 * AFLOWDataProvider::entry_url_from_aflux builds the entry file's URL from
 * the AFLUX answer (test_aflow_live_fixture.cpp has the answer the service
 * gives today); here the answers that give no URL are refused, each saying
 * why.
 *
 * AflowReader::read asks its provider for the entry and refuses an answer
 * whose HTTP status is not 200. A stub IMaterialsDataProvider stands in for
 * the service: QueryByID records the ID it is asked for and returns the next
 * RawResponse of a script, so the reader's handling of the provider's
 * answer is tested without cURL. The reasons in the scripts are the
 * provider's own: the 404 that fetchAndParseAflowByAuid returns when the
 * AFLUX answer gives no entry URL, and the 500 that httpGet returns on a
 * cURL error (the text of curl_easy_strerror(CURLE_COULDNT_RESOLVE_HOST) in
 * libcurl 8.9.1). The reader is built on the stub with
 * AflowReader(std::unique_ptr<IMaterialsDataProvider>), the pointer the
 * reader already holds its provider in; the constructor is detected, so
 * that the file compiles before it exists. */
#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/data_providers/aflow_data_provider.h>
#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/aflow_reader.h>

#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using tadah::core::PeriodicTable;
using tadah::mlip::AFLOWDataProvider;
using tadah::mlip::AflowReader;
using tadah::mlip::IMaterialsDataProvider;
using tadah::mlip::RawResponse;
using tadah::mlip::Structure;

namespace {

using ProviderPtr = std::unique_ptr<IMaterialsDataProvider>;

/* Whether R can be built on a provider: AflowReader(ProviderPtr). */
template <class R>
constexpr bool takes_provider = std::is_constructible<R, ProviderPtr>::value;

/* What the stub answers, in order, and the IDs it was asked for. Shared
 * with the test, since the stub itself is moved into the reader. */
struct Script {
  std::deque<RawResponse> answers;
  std::vector<std::string> asked;
};

class StubProvider : public IMaterialsDataProvider {
public:
  explicit StubProvider(std::shared_ptr<Script> script)
      : m_script(std::move(script)) {}

  RawResponse QueryByID(const std::string &materialID) override {
    m_script->asked.push_back(materialID);
    if (m_script->answers.empty()) {
      // Not a std::runtime_error: the reader must not take it for a refusal.
      throw std::logic_error("StubProvider: no answer left for '" +
                             materialID + "'.");
    }
    RawResponse rr = m_script->answers.front();
    m_script->answers.pop_front();
    return rr;
  }

  RawResponse QueryByFormula(const std::string &) override {
    return {501, "StubProvider: no formula queries."};
  }

private:
  std::shared_ptr<Script> m_script;
};

/* One read: whether it threw a std::runtime_error, its message, and the
 * reader's structure afterwards. */
struct Outcome {
  bool threw = false;
  std::string error;
  Structure st;
};

/* R::read(id) for each of `ids` in turn, on one R built on a stub that
 * answers with `script`. Empty when R cannot be built on a provider. */
template <class R>
std::vector<Outcome> read_through(const std::shared_ptr<Script> &script,
                                  const std::vector<std::string> &ids) {
  std::vector<Outcome> out;
  if constexpr (takes_provider<R>) {
    R reader(ProviderPtr(std::make_unique<StubProvider>(script)));
    for (const std::string &id : ids) {
      Outcome o;
      try {
        reader.read(id);
      } catch (const std::runtime_error &e) {
        o.threw = true;
        o.error = e.what();
      }
      o.st = reader.getStructure();
      out.push_back(o);
    }
  }
  return out;
}

std::shared_ptr<Script> script_of(std::vector<RawResponse> answers) {
  auto s = std::make_shared<Script>();
  for (RawResponse &rr : answers) s->answers.push_back(std::move(rr));
  return s;
}

/* Rock-salt NaCl as aflowlib.json writes it (typed fields); the numbers
 * are those of test_aflow_reader_typed.cpp, not values aflowlib serves. */
const std::string kNaclAuid = "aflow:0000000000000003";
const std::string kNaclEntry =
    R"({"auid":"aflow:0000000000000003","compound":"Cl1Na1",)"
    R"("species":["Cl","Na"],"composition":[1,1],)"
    R"("volume_cell":44.85154,"density":2.163728,)"
    R"("geometry":[3.98808224589,3.98808224589,3.98808224589,60,60,60],)"
    R"("positions_fractional":[[0,0,0],[0.5,0.5,0.5]]})";

}  // namespace

/* Pins: an AFLUX answer that gives no entry URL is refused by
 * AFLOWDataProvider::entry_url_from_aflux with a std::runtime_error saying
 * why:
 *   "{}", "[]" and "not json" (no array, or an empty one):
 *     'AFLOW: the AFLUX answer is not a non-empty JSON array.'
 *   [3], [{"x":1}] and [{"aurl":3}] (no "aurl" string in the first entry):
 *     'AFLOW: the first entry of the AFLUX answer has no "aurl"
 *     string.'
 *   [{"aurl":"nocolon"}], [{"aurl":":AFLOWDATA/x"}] and
 *   [{"aurl":"aflowlib.duke.edu:"}] (no server or no path):
 *     "AFLOW: aurl '<aurl>' in the AFLUX answer is not
 *     server:PATH."
 * Before the fix: this already holds on this branch, where
 * entry_url_from_aflux was added. On v1-dev the checks were inline in
 * fetchAndParseAflowByAuid, with no function to test: "{}", "[]" and "not
 * json" gave 404 "AFLOWDataProvider: 'aflux' JSON not an array or is
 * empty.", [3] and [{"x":1}] 404 "AFLOWDataProvider: 'aurl' field not found
 * in JSON.", [{"aurl":3}] threw nlohmann::json::type_error 302 out of
 * QueryByID, and an aurl without a colon was requested as it stood, with
 * "?format=json" appended.
 * Before the fix that names the database: each message began
 * "AFLOWDataProvider: ": the checks fail. */
TEST_CASE("AFLOW provider: an AFLUX answer that gives no entry URL is "
          "refused, saying why",
          "[aflow][data_providers]") {
  const std::string not_array =
      "AFLOW: the AFLUX answer is not a non-empty JSON array.";
  const std::string no_aurl =
      "AFLOW: the first entry of the AFLUX answer has no \"aurl\" string.";
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"{}", not_array},
      {"[]", not_array},
      {"not json", not_array},
      {"[3]", no_aurl},
      {R"([{"x":1}])", no_aurl},
      {R"([{"aurl":3}])", no_aurl},
      {R"([{"aurl":"nocolon"}])",
       "AFLOW: aurl 'nocolon' in the AFLUX answer is not server:PATH."},
      {R"([{"aurl":":AFLOWDATA/x"}])",
       "AFLOW: aurl ':AFLOWDATA/x' in the AFLUX answer is not server:PATH."},
      {R"([{"aurl":"aflowlib.duke.edu:"}])",
       "AFLOW: aurl 'aflowlib.duke.edu:' in the AFLUX answer is not "
       "server:PATH."},
  };
  for (const auto &b : bad) {
    INFO("AFLUX answer: " << b.first);
    CHECK_THROWS_AS(AFLOWDataProvider::entry_url_from_aflux(b.first),
                    std::runtime_error);
    CHECK_THROWS_WITH(AFLOWDataProvider::entry_url_from_aflux(b.first),
                      Catch::Contains(b.second));
  }
}

/* Pins: when the provider answers a status other than 200, read(auid)
 * throws a std::runtime_error that gives the status, the AUID and the
 * provider's reason (its RawResponse::response), so that the user sees why
 * the entry could not be fetched. For "aflow:0000000000000009" and the
 * answers
 *   404 "AFLOW: the AFLUX answer is not a non-empty JSON array."
 *   500 "AFLOW: cURL error: Could not resolve hostname"
 * the message holds "HTTP 404" (or "HTTP 500"), the AUID and the reason
 * word for word. The provider is asked once, for that AUID.
 * Before the fix: AflowReader cannot be built on a provider (it makes its
 * own AFLOWDataProvider), so takes_provider<AflowReader> is false and no
 * read is made; and fetchAndParseAflowByAuid throws only
 * "AflowReader: HTTP <status> for AUID '<auid>'.", dropping the
 * provider's reason. */
TEST_CASE("AFLOW reader: a failed query gives the AUID and the provider's "
          "reason",
          "[aflow][structure_readers][data_providers]") {
  CHECK(takes_provider<AflowReader>);
  const std::string auid = "aflow:0000000000000009";
  const std::vector<RawResponse> answers = {
      {404, "AFLOW: the AFLUX answer is not a non-empty JSON array."},
      {500, "AFLOW: cURL error: Could not resolve hostname"},
  };
  for (const RawResponse &answer : answers) {
    INFO("HTTP " << answer.httpStatus << ": " << answer.response);
    const auto script = script_of({answer});
    const std::vector<Outcome> got = read_through<AflowReader>(script, {auid});
    REQUIRE(got.size() == 1);
    INFO("message: " << got[0].error);
    CHECK(got[0].threw);
    CHECK_THAT(got[0].error,
               Catch::Contains("HTTP " + std::to_string(answer.httpStatus)));
    CHECK_THAT(got[0].error, Catch::Contains(auid));
    CHECK_THAT(got[0].error, Catch::Contains(answer.response));
    CHECK(script->asked == std::vector<std::string>{auid});
  }
}

/* Pins: a long reason is cut short in the message: for a 404 whose body is
 * a 4107-character HTML page, the message still gives "HTTP 404", the AUID
 * and the first 100 characters of the page, and is shorter than 1000
 * characters.
 * Before the fix: AflowReader cannot be built on a provider, so no read is
 * made (and the message would have no part of the page at all). */
TEST_CASE("AFLOW reader: a long reason from the provider is cut short",
          "[aflow][structure_readers][data_providers]") {
  CHECK(takes_provider<AflowReader>);
  const std::string auid = "aflow:0000000000000009";
  const std::string page =
      "<!DOCTYPE html><html><head><title>404 Not Found</title></head>"
      "<body><h1>Not Found</h1><p>" +
      std::string(4000, 'x') + "</p></body></html>";
  REQUIRE(page.size() > 4000);
  const auto script = script_of({{404, page}});
  const std::vector<Outcome> got = read_through<AflowReader>(script, {auid});
  REQUIRE(got.size() == 1);
  INFO("message: " << got[0].error);
  CHECK(got[0].threw);
  CHECK_THAT(got[0].error, Catch::Contains("HTTP 404"));
  CHECK_THAT(got[0].error, Catch::Contains(auid));
  CHECK_THAT(got[0].error, Catch::Contains(page.substr(0, 100)));
  CHECK(got[0].error.size() < 1000);
}

/* Pins: a reader built on a provider reads what the provider returns: for a
 * 200 answer whose body is the typed NaCl entry, read("aflow:0000000000000003")
 * asks the provider for that AUID and gives exactly the structure
 * parseAflowJson gives for the body (label, cell, atoms); a later read whose
 * answer is 404 throws a std::runtime_error and leaves that structure as it
 * was.
 * Before the fix: AflowReader cannot be built on a provider, so no read is
 * made and the test has nothing to compare. */
TEST_CASE("AFLOW reader: the entry a provider returns is read",
          "[aflow][structure_readers][data_providers]") {
  PeriodicTable::initialize();
  CHECK(takes_provider<AflowReader>);
  AflowReader direct;
  REQUIRE_NOTHROW(direct.parseAflowJson(kNaclEntry));
  const Structure want = direct.getStructure();
  REQUIRE(want.natoms() == 2);

  const std::string missing = "aflow:000000000000000a";
  const auto script = script_of(
      {{200, kNaclEntry},
       {404, "AFLOW: the AFLUX answer is not a non-empty JSON array."}});
  const std::vector<Outcome> got =
      read_through<AflowReader>(script, {kNaclAuid, missing});
  REQUIRE(got.size() == 2);
  CHECK(script->asked == std::vector<std::string>{kNaclAuid, missing});

  INFO("first read: " << got[0].error);
  CHECK_FALSE(got[0].threw);
  INFO("second read: " << got[1].error);
  CHECK(got[1].threw);
  CHECK_THAT(got[1].error, Catch::Contains(missing));
  for (const Outcome &o : got) {
    CHECK(o.st.label == want.label);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) CHECK(o.st.cell(i, k) == want.cell(i, k));
    REQUIRE(o.st.natoms() == want.natoms());
    for (size_t n = 0; n < want.natoms(); ++n) {
      CHECK(o.st.atoms[n].symbol == want.atoms[n].symbol);
      for (int k = 0; k < 3; ++k)
        CHECK(o.st.atoms[n].position[k] == want.atoms[n].position[k]);
    }
  }
}

/* Pins: an AflowReader built on a null provider is refused at once
 * (std::invalid_argument naming the provider), not when it reads.
 * Before the fix: AflowReader had no constructor taking a provider.
 * Pins (the message): "internal error: the AFLOW reader has no data
 * provider". Before the fix that names the database: "AflowReader: the data
 * provider is null.": the check fails. */
TEST_CASE("AFLOW reader: a null provider is refused",
          "[structure_readers][aflow]") {
  CHECK(takes_provider<AflowReader>);
  if constexpr (takes_provider<AflowReader>) {
    CHECK_THROWS_AS(AflowReader(ProviderPtr{}), std::invalid_argument);
    CHECK_THROWS_WITH(AflowReader(ProviderPtr{}),
                      "internal error: the AFLOW reader has no data provider");
  }
}
