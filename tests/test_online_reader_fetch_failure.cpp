/* NomadReader, MaterialsProjectReader and CODReader without the network.
 *
 * Each reader asks its IMaterialsDataProvider for the entry and refuses an
 * answer whose HTTP status is not 200. As AflowReader does
 * (test_aflow_provider.cpp), the refusal gives the status, the ID and the
 * provider's reason (RawResponse::response: the provider's own text for a
 * cURL error, else the server's body), cut to 300 characters:
 *   "<Reader>: HTTP <status> for <what> '<id>': <reason>"
 * and, with no reason, "<Reader>: HTTP <status> for <what> '<id>'.", where
 * <what> is "entry" (NOMAD), "material" (Materials Project) or "COD ID". The
 * reader adds nothing else: no request URL, no header, no API key.
 *
 * A stub provider stands in for the service: QueryByID records the ID it is
 * asked for and returns the next RawResponse of a script. The cURL reasons
 * are the providers' own wording (nomad_data_provider.cpp,
 * materials_project_data_provider.cpp, cod_data_provider.cpp) with the text
 * of curl_easy_strerror(CURLE_COULDNT_CONNECT); the 404 bodies are short
 * stand-ins of the shape each service answers with (JSON, or an HTML page
 * for COD), not captured answers. Each reader is built on the stub with
 * R(std::unique_ptr<IMaterialsDataProvider>), as AflowReader is; the
 * constructor is detected, so that the file compiles before it exists.
 *
 * An entry the service does return is parsed; NomadReader and
 * MaterialsProjectReader name the entry or material (the ID given to
 * read()) and the site when a site's element is one Tadah does not know,
 * and a refused entry leaves the reader's structure as it was. */
#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/cod_reader.h>
#include <tadah/mlip/structure_readers/materials_project_reader.h>
#include <tadah/mlip/structure_readers/nomad_reader.h>

#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using tadah::mlip::CODReader;
using tadah::mlip::IMaterialsDataProvider;
using tadah::mlip::MaterialsProjectReader;
using tadah::mlip::NomadReader;
using tadah::mlip::RawResponse;
using tadah::mlip::Structure;

namespace {

using ProviderPtr = std::unique_ptr<IMaterialsDataProvider>;

/* Whether R can be built on a provider: R(ProviderPtr). */
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

/* One reader's failed queries: the ID given to read(), the ID the provider
 * must be asked for and the message names, and two answers (a cURL error
 * and a 404 with a short body). */
struct FailCase {
  std::string reader;   // "NomadReader"
  std::string what;     // "entry"
  std::string read_id;  // what read() is given
  std::string id;       // what the provider is asked for and the message names
  RawResponse curl_error;
  RawResponse not_found;
};

/* A 404 whose body is a 4107-character HTML page. */
std::string long_page() {
  return "<!DOCTYPE html><html><head><title>404 Not Found</title></head>"
         "<body><h1>Not Found</h1><p>" +
         std::string(4000, 'x') + "</p></body></html>";
}

/* The checks every reader shares (see the test cases below). */
template <class R>
void check_failed_queries(const FailCase &fc) {
  CHECK(takes_provider<R>);
  for (const RawResponse &answer : {fc.curl_error, fc.not_found}) {
    INFO(fc.reader << " read('" << fc.read_id << "'), HTTP "
                   << answer.httpStatus << ": " << answer.response);
    const auto script = script_of({answer});
    const std::vector<Outcome> got = read_through<R>(script, {fc.read_id});
    REQUIRE(got.size() == 1);
    INFO("message: " << got[0].error);
    CHECK(got[0].threw);
    CHECK_THAT(got[0].error,
               Catch::Contains("HTTP " + std::to_string(answer.httpStatus)));
    CHECK_THAT(got[0].error, Catch::Contains("'" + fc.id + "'"));
    CHECK_THAT(got[0].error, Catch::Contains(answer.response));
    CHECK(got[0].error == fc.reader + ": HTTP " +
                              std::to_string(answer.httpStatus) + " for " +
                              fc.what + " '" + fc.id + "': " +
                              answer.response);
    CHECK(script->asked == std::vector<std::string>{fc.id});
  }

  // No reason: the message ends after the ID.
  {
    const auto script = script_of({{404, ""}});
    const std::vector<Outcome> got = read_through<R>(script, {fc.read_id});
    REQUIRE(got.size() == 1);
    CHECK(got[0].threw);
    CHECK(got[0].error ==
          fc.reader + ": HTTP 404 for " + fc.what + " '" + fc.id + "'.");
  }

  // A long reason is cut to 300 characters, and its length is given.
  {
    const std::string page = long_page();
    REQUIRE(page.size() == 4107);
    const auto script = script_of({{404, page}});
    const std::vector<Outcome> got = read_through<R>(script, {fc.read_id});
    REQUIRE(got.size() == 1);
    INFO("message: " << got[0].error);
    CHECK(got[0].threw);
    CHECK_THAT(got[0].error, Catch::Contains("HTTP 404"));
    CHECK_THAT(got[0].error, Catch::Contains("'" + fc.id + "'"));
    CHECK_THAT(got[0].error, Catch::Contains(page.substr(0, 100)));
    CHECK(got[0].error.size() < 1000);
    CHECK(got[0].error == fc.reader + ": HTTP 404 for " + fc.what + " '" +
                              fc.id + "': " + page.substr(0, 300) +
                              " ... (4107 characters)");
  }
}

/* A reader built on a null provider is refused at once. */
template <class R>
void check_null_provider_refused(const std::string &reader) {
  INFO(reader);
  CHECK(takes_provider<R>);
  if constexpr (takes_provider<R>) {
    CHECK_THROWS_AS(R(ProviderPtr{}), std::invalid_argument);
    CHECK_THROWS_WITH(R(ProviderPtr{}), Catch::Contains("provider is null"));
    CHECK_THROWS_WITH(R(ProviderPtr{}),
                      reader + ": the data provider is null.");
  }
}

}  // namespace

/* Pins: NomadReader::read("zyXabc123zyXabc123zyXab") on a provider that
 * answers
 *   500 "NomadDataProvider::httpPost cURL error: Could not connect to server"
 *   404 '{"detail":"The entry with the given id does not exist."}'
 * throws exactly "NomadReader: HTTP <status> for entry '<id>': <reason>";
 * the provider is asked once, for that ID; an empty reason gives "... for
 * entry '<id>'."; a 4107-character page is cut to its first 300 characters
 * followed by " ... (4107 characters)".
 * Before the fix: NomadReader cannot be built on a provider (it makes its
 * own NomadDataProvider), so takes_provider<NomadReader> is false and no
 * read is made (REQUIRE(got.size() == 1) fails); and the message was
 * "NomadReader: HTTP 500 fetching entry '<id>'.", without the reason. */
TEST_CASE("NOMAD reader: a failed query gives the status, the entry ID and "
          "the provider's reason",
          "[nomad][structure_readers][data_providers]") {
  check_failed_queries<NomadReader>(
      {"NomadReader", "entry", "zyXabc123zyXabc123zyXab",
       "zyXabc123zyXabc123zyXab",
       {500, "NomadDataProvider::httpPost cURL error: Could not connect to "
             "server"},
       {404, R"({"detail":"The entry with the given id does not exist."})"}});
}

/* Pins: the same for MaterialsProjectReader::read("mp-149"):
 *   500 "MaterialsProjectDataProvider: cURL error: Could not connect to
 *       server"
 *   404 '{"detail":"Not Found"}'
 * give exactly "MaterialsProjectReader: HTTP <status> for material
 * 'mp-149': <reason>" (no URL, header or API key), asking the provider once
 * for "mp-149"; empty and long reasons as for NOMAD.
 * Before the fix: MaterialsProjectReader can be built only from an API key,
 * so no read is made; the message was "MaterialsProjectReader: HTTP <status>
 * retrieving 'mp-149'.". */
TEST_CASE("Materials Project reader: a failed query gives the status, the "
          "material ID and the provider's reason",
          "[materials_project][structure_readers][data_providers]") {
  check_failed_queries<MaterialsProjectReader>(
      {"MaterialsProjectReader", "material", "mp-149", "mp-149",
       {500, "MaterialsProjectDataProvider: cURL error: Could not connect to "
             "server"},
       {404, R"({"detail":"Not Found"})"}});
}

/* Pins: the same for CODReader, which asks for the COD ID without the
 * "cod-" prefix and names it as COD does: read("cod-1534932"),
 * read("1534932") and read("https://www.crystallography.net/cod/1534932.cif")
 * each ask the provider once for "1534932" and, for
 *   500 "CODDataProvider: cURL error: Could not connect to server"
 *   404 "<html><body><h1>404 Not Found</h1></body></html>"
 * throw exactly "CODReader: HTTP <status> for COD ID '1534932': <reason>";
 * empty and long reasons as for NOMAD.
 * Before the fix: CODReader cannot be built on a provider, so no read is
 * made; the message was "CODReader: CODDataProvider returned HTTP <status>
 * for ID '1534932'.", without the reason. */
TEST_CASE("COD reader: a failed query gives the status, the COD ID and the "
          "provider's reason",
          "[cod][structure_readers][data_providers]") {
  for (const std::string read_id :
       {"cod-1534932", "1534932",
        "https://www.crystallography.net/cod/1534932.cif"}) {
    check_failed_queries<CODReader>(
        {"CODReader", "COD ID", read_id, "1534932",
         {500, "CODDataProvider: cURL error: Could not connect to server"},
         {404, "<html><body><h1>404 Not Found</h1></body></html>"}});
  }
}

/* Pins: each reader built on a null provider is refused at once
 * (std::invalid_argument, "<Reader>: the data provider is null."), not when
 * it reads.
 * Before the fix: none of the three had a constructor taking a provider. */
TEST_CASE("NOMAD, Materials Project and COD readers: a null provider is "
          "refused",
          "[structure_readers][data_providers]") {
  check_null_provider_refused<NomadReader>("NomadReader");
  check_null_provider_refused<MaterialsProjectReader>("MaterialsProjectReader");
  check_null_provider_refused<CODReader>("CODReader");
}

namespace {

/* bcc Fe as COD serves a CIF: a block named after the COD ID, a = 2.8665 A,
 * H-M 'I m -3 m', Fe on 2a; `occupancy` the site's occupancy column. */
std::string cod_fe_cif(const std::string &block, const std::string &occupancy) {
  return "data_" + block +
         "\n_cell_length_a 2.8665\n_cell_length_b 2.8665\n"
         "_cell_length_c 2.8665\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
         "_cell_angle_gamma 90\n_symmetry_space_group_name_H-M 'I m -3 m'\n"
         "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
         "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
         "_atom_site_occupancy\nFe1 Fe 0 0 0 " +
         occupancy + "\n";
}

}  // namespace

/* Pins: CODReader names the COD entry, not the CIF data block, in its
 * errors and its label. Through a stub provider:
 *   - read("cod-1534932") of a CIF whose block is 'data_global' and whose
 *     site Fe1 has occupancy 0.5 throws "COD entry 1534932: site Fe1 has
 *     occupancy 0.5; ..." (no "CIF text", no "data_global"), and keeps
 *     the previous structure;
 *   - read("cod-1534932") of bcc Fe in a block 'data_1534932' gives the
 *     label "COD entry 1534932 | Symmetry: I m -3 m | Volume: 23.553521"
 *     (the volume 2.8665^3 as std::to_string writes it) and 2 atoms.
 * Before the fix (with the provider constructor already in place): the
 * label was "COD entry: cod-1534932 |  | Symmetry: I m -3 m | Volume:
 * 23.553521" (the read() argument, and an empty field) and the error
 * named the block, "CIF text (data_global): site Fe1 has occupancy 0.5;
 * ...". */
TEST_CASE("COD reader: errors and the label name the COD entry",
          "[cod][structure_readers][cif]") {
  tadah::core::PeriodicTable::initialize();
  CHECK(takes_provider<CODReader>);
  const auto script = script_of(
      {{200, cod_fe_cif("1534932", "1.0")}, {200, cod_fe_cif("global", "0.5")}});
  const std::vector<Outcome> got =
      read_through<CODReader>(script, {"cod-1534932", "cod-1534932"});
  REQUIRE(got.size() == 2);
  CHECK(script->asked == std::vector<std::string>{"1534932", "1534932"});

  INFO("first read: " << got[0].error);
  CHECK_FALSE(got[0].threw);
  CHECK(got[0].st.label ==
        "COD entry 1534932 | Symmetry: I m -3 m | Volume: 23.553521");
  CHECK(got[0].st.natoms() == 2);

  INFO("second read: " << got[1].error);
  CHECK(got[1].threw);
  CHECK_THAT(got[1].error,
             Catch::Contains("COD entry 1534932: site Fe1 has occupancy 0.5"));
  CHECK_THAT(got[1].error, !Catch::Contains("CIF text"));
  CHECK_THAT(got[1].error, !Catch::Contains("data_global"));
  CHECK(got[1].st.label == got[0].st.label);
  CHECK(got[1].st.natoms() == 2);
}

/* Pins: CODReader names the COD entry and the site when the CIF has a site
 * whose element Tadah does not know (deuterium, D, common in COD's
 * neutron-diffraction entries): read("cod-1534932") of a P 1 CIF in a
 * block 'data_1534932' with sites Fe1 (Fe) and D1 (D) throws exactly
 *   "COD entry 1534932: site D1 has element 'D', which Tadah does not
 *    know: its elements are H to Og."
 * and keeps the previous structure (bcc Fe, 2 atoms).
 * Before the fix: the periodic table's own error escaped, "Unknown symbol:
 * D", naming neither the COD entry nor the site. */
TEST_CASE("COD reader: a site whose element Tadah does not know is refused, "
          "naming the COD entry and the site",
          "[cod][structure_readers][cif]") {
  tadah::core::PeriodicTable::initialize();
  CHECK(takes_provider<CODReader>);
  const std::string deuterium =
      "data_1534932\n_cell_length_a 3.0\n_cell_length_b 3.0\n"
      "_cell_length_c 3.0\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
      "_cell_angle_gamma 90\n_symmetry_space_group_name_H-M 'P 1'\n"
      "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
      "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
      "Fe1 Fe 0 0 0\nD1 D 0.5 0.5 0.5\n";
  const auto script = script_of(
      {{200, cod_fe_cif("1534932", "1.0")}, {200, deuterium}});
  const std::vector<Outcome> got =
      read_through<CODReader>(script, {"cod-1534932", "cod-1534932"});
  REQUIRE(got.size() == 2);
  CHECK(script->asked == std::vector<std::string>{"1534932", "1534932"});
  CHECK_FALSE(got[0].threw);
  INFO("second read: " << got[1].error);
  CHECK(got[1].threw);
  CHECK(got[1].error ==
        "COD entry 1534932: site D1 has element 'D', which Tadah does not "
        "know: its elements are H to Og.");
  CHECK(got[1].st.label == got[0].st.label);
  CHECK(got[1].st.natoms() == 2);
}

namespace {

/* A NOMAD entry as the archive query answers it, cut to the fields
 * NomadReader reads: bcc Fe, a = 2.8665 A, two sites whose species are
 * `first` and `second`. */
std::string nomad_fe(const std::string &entry_id, const std::string &first,
                     const std::string &second) {
  return R"({"data":{"entry_id":")" + entry_id +
         R"(","archive":{"results":{"material":{"symmetry":)"
         R"({"space_group_symbol":"Im-3m"},"topology":[{"cell":)"
         R"({"volume":2.355e-29,"mass_density":7874.0}}]}},)"
         R"("metadata":{"optimade":{"lattice_vectors":)"
         R"([[2.8665,0,0],[0,2.8665,0],[0,0,2.8665]],)"
         R"("cartesian_site_positions":[[0,0,0],[1.43325,1.43325,1.43325]],)"
         R"("species_at_sites":[")" +
         first + R"(",")" + second + R"("]}}}}})";
}

/* A Materials Project "materials/core" answer, cut to the fields
 * MaterialsProjectReader reads: bcc Fe, a = 2.8665 A, two sites whose
 * element (and label) are `first` and `second`. */
std::string mp_fe(const std::string &material_id, const std::string &first,
                  const std::string &second) {
  auto site = [](const std::string &el, const std::string &abc) {
    return R"({"species":[{"element":")" + el + R"(","occu":1}],"abc":)" +
           abc + R"(,"label":")" + el + R"("})";
  };
  return R"({"data":[{"material_id":")" + material_id +
         R"(","symmetry":{"symbol":"Im-3m"},"volume":23.553521,)"
         R"("density":7.87,"elements":["Fe"],"structure":{"lattice":)"
         R"({"matrix":[[2.8665,0,0],[0,2.8665,0],[0,0,2.8665]]},"sites":[)" +
         site(first, "[0,0,0]") + "," + site(second, "[0.5,0.5,0.5]") +
         "]}}]}";
}

}  // namespace

/* Pins: NomadReader names the entry, as the ID given to read(), in its
 * errors: through a stub provider,
 *   - read("zyXFe") of bcc Fe gives 2 atoms; reading it again gives 2 atoms
 *     (the structure is replaced, not added to);
 *   - read("zyXD") of an entry whose second site has species 'D'
 *     (deuterium, which Tadah has not) throws exactly
 *       NomadReader: entry 'zyXD': site 2 has species 'D', which is not an
 *       element symbol Tadah knows (H to Og).
 *     and keeps the previous structure (label and 2 atoms);
 *   - read("zyXbad") of a body that is not JSON throws a std::runtime_error
 *     that starts "NomadReader: entry 'zyXbad': ".
 * Before the fix: the periodic table's error escaped, 'Unknown symbol: D',
 * naming neither the entry nor the site; the second read of bcc Fe appended
 * its atoms to the first ones (4 atoms), the refused entry had set its label
 * and appended its first atom (5), and the JSON parser's own
 * exception (nlohmann::json::parse_error, not a std::runtime_error) escaped
 * read_through, naming no entry. */
TEST_CASE("NOMAD reader: a site whose element Tadah does not know is "
          "refused, naming the entry and the site",
          "[nomad][structure_readers][data_providers]") {
  tadah::core::PeriodicTable::initialize();
  CHECK(takes_provider<NomadReader>);
  const auto script =
      script_of({{200, nomad_fe("zyXFe", "Fe", "Fe")},
                 {200, nomad_fe("zyXFe", "Fe", "Fe")},
                 {200, nomad_fe("zyXD", "Fe", "D")}});
  std::vector<Outcome> got;
  REQUIRE_NOTHROW(got = read_through<NomadReader>(
                      script, {"zyXFe", "zyXFe", "zyXD"}));
  REQUIRE(got.size() == 3);
  INFO("first read: " << got[0].error);
  CHECK_FALSE(got[0].threw);
  CHECK(got[0].st.natoms() == 2);
  INFO("second read: " << got[1].error);
  CHECK_FALSE(got[1].threw);
  CHECK(got[1].st.natoms() == 2);
  INFO("third read: " << got[2].error);
  CHECK(got[2].threw);
  CHECK(got[2].error ==
        "NomadReader: entry 'zyXD': site 2 has species 'D', which is not an "
        "element symbol Tadah knows (H to Og).");
  CHECK(got[2].st.label == got[0].st.label);
  CHECK(got[2].st.natoms() == 2);
  // A body that is not JSON: the parser's error, named by the reader.
  const auto bad_script = script_of({{200, "<html>not JSON</html>"}});
  std::vector<Outcome> bad;
  REQUIRE_NOTHROW(bad = read_through<NomadReader>(bad_script, {"zyXbad"}));
  REQUIRE(bad.size() == 1);
  INFO("not JSON: " << bad[0].error);
  CHECK(bad[0].threw);
  CHECK(bad[0].error.rfind("NomadReader: entry 'zyXbad': ", 0) == 0);
}

/* Pins: MaterialsProjectReader names the material, as the ID given to
 * read(), in its errors: through a stub provider,
 *   - read("mp-13") of bcc Fe gives 2 atoms; reading it again gives 2
 *     atoms;
 *   - read("mp-xx") of a material whose second site has element 'Xx'
 *     throws exactly
 *       MaterialsProjectReader: material 'mp-xx': site 2 has element 'Xx',
 *       which is not an element symbol Tadah knows (H to Og).
 *     and keeps the previous structure (label and 2 atoms);
 *   - read("mp-bad") of a body that is not JSON throws a std::runtime_error
 *     that starts "MaterialsProjectReader: material 'mp-bad': ".
 * Before the fix: 'Unknown symbol: Xx' escaped, naming neither the material
 * nor the site; the second read of bcc Fe appended its atoms (4 atoms), the
 * refused material had set its label and appended its first atom (5), and
 * the JSON parser's own exception (nlohmann::json::parse_error, not a
 * std::runtime_error) escaped read_through, naming no material. */
TEST_CASE("Materials Project reader: a site whose element Tadah does not "
          "know is refused, naming the material and the site",
          "[materials_project][structure_readers][data_providers]") {
  tadah::core::PeriodicTable::initialize();
  CHECK(takes_provider<MaterialsProjectReader>);
  const auto script = script_of({{200, mp_fe("mp-13", "Fe", "Fe")},
                                 {200, mp_fe("mp-13", "Fe", "Fe")},
                                 {200, mp_fe("mp-xx", "Fe", "Xx")}});
  std::vector<Outcome> got;
  REQUIRE_NOTHROW(got = read_through<MaterialsProjectReader>(
                      script, {"mp-13", "mp-13", "mp-xx"}));
  REQUIRE(got.size() == 3);
  INFO("first read: " << got[0].error);
  CHECK_FALSE(got[0].threw);
  CHECK(got[0].st.natoms() == 2);
  INFO("second read: " << got[1].error);
  CHECK_FALSE(got[1].threw);
  CHECK(got[1].st.natoms() == 2);
  INFO("third read: " << got[2].error);
  CHECK(got[2].threw);
  CHECK(got[2].error ==
        "MaterialsProjectReader: material 'mp-xx': site 2 has element 'Xx', "
        "which is not an element symbol Tadah knows (H to Og).");
  CHECK(got[2].st.label == got[0].st.label);
  CHECK(got[2].st.natoms() == 2);
  // A body that is not JSON: the parser's error, named by the reader.
  const auto bad_script = script_of({{200, "<html>not JSON</html>"}});
  std::vector<Outcome> bad;
  REQUIRE_NOTHROW(bad = read_through<MaterialsProjectReader>(bad_script, {"mp-bad"}));
  REQUIRE(bad.size() == 1);
  INFO("not JSON: " << bad[0].error);
  CHECK(bad[0].threw);
  CHECK(bad[0].error.rfind("MaterialsProjectReader: material 'mp-bad': ", 0) == 0);
}
