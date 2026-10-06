// AflowReader::parseAflowJson on the two encodings of an aflowlib entry.
//
// The string form is the one the reader was first written for: every field
// a JSON string, lists comma-separated and sites separated by ';' (the
// layout of the AFLOW REST API's text output). The typed form is the one
// AFLOW's JSON output uses (aflowlib.json, which AFLOWDataProvider fetches):
//   geometry              [a, b, c, alpha, beta, gamma]  six numbers
//   species               ["Cl", "Na"]                   array of strings
//   composition           [1, 1]                         array of numbers
//   positions_fractional  [[0,0,0], [0.5,0.5,0.5]]       array of 3-arrays
//   volume_cell, density  numbers
//   compound, auid        strings, the same in both forms
// These typed shapes were checked against the live Ta1_ICSD_76152 entry,
// saved on 2026-10-06 with MK's approval (test_aflow_live_fixture.cpp).
//
// The entries are written by hand: bcc Ta with a_c = 3.3058 A (the 1-atom
// primitive cell, the shape of the ICSD_WEB/BCC/Ta1 entry, and the 2-atom
// conventional cell) and rock-salt NaCl with a_c = 5.64 A (two species).
// Their numbers fit these lattice constants; they are not the values
// aflowlib serves. The keys the reader does not use (aurl, natoms,
// species_pp, positions_cartesian, Egap) are there to show that they are
// ignored in either form.

#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/aflow_reader.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::PeriodicTable;
using namespace tadah::mlip;

namespace {

// One field of an entry, as JSON value text in each form.
struct Field {
  std::string key;
  std::string str;    // the string form
  std::string typed;  // the typed form
};

struct Entry {
  std::string name;
  std::string auid;
  std::vector<std::string> symbols;  // the atoms expected, in order
  std::string label;                 // the label expected
  std::vector<Field> fields;
};

enum class Form { String, Typed };

// The fields whose two forms differ.
const std::vector<std::string> kTypedKeys = {
    "species", "composition", "volume_cell",
    "density", "geometry",    "positions_fractional"};

std::string q(const std::string &s) { return "\"" + s + "\""; }

// A field that is a JSON string in both forms.
Field same(const std::string &key, const std::string &value) {
  return {key, q(value), q(value)};
}

std::vector<Entry> entries() {
  const std::string ta_p = "2.8629068";    // sqrt(3)/2 a_c
  const std::string bcc = "109.4712206";   // acos(-1/3) in degrees
  const std::string ta_geom =
      ta_p + "," + ta_p + "," + ta_p + "," + bcc + "," + bcc + "," + bcc;
  const std::string ta_c = "3.3058";
  const std::string ta_conv = ta_c + "," + ta_c + "," + ta_c + ",90,90,90";
  const std::string nacl_p = "3.98808224589";  // 5.64/sqrt(2)
  const std::string nacl_geom =
      nacl_p + "," + nacl_p + "," + nacl_p + ",60,60,60";
  return {
      {"bcc Ta, primitive cell, 1 atom",
       "aflow:0000000000000001",
       {"Ta"},
       "aflow:0000000000000001 | Ta1 | Volume: 18.06341 | Density: 16.634236",
       {same("aurl", "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152"),
        same("auid", "aflow:0000000000000001"),
        same("compound", "Ta1"),
        {"natoms", q("1"), "1"},
        {"species", q("Ta"), R"(["Ta"])"},
        {"species_pp", q("Ta_pv"), R"(["Ta_pv"])"},
        {"composition", q("1"), "[1]"},
        {"volume_cell", q("18.06341"), "18.06341"},
        {"density", q("16.634236"), "16.634236"},
        {"geometry", q(ta_geom), "[" + ta_geom + "]"},
        {"positions_cartesian", q("0,0,0"), "[[0,0,0]]"},
        {"positions_fractional", q("0,0,0"), "[[0,0,0]]"},
        {"Egap", q("0"), "0"}}},
      {"bcc Ta, conventional cell, 2 atoms",
       "aflow:0000000000000002",
       {"Ta", "Ta"},
       "aflow:0000000000000002 | Ta2 | Volume: 36.12682 | Density: 16.634236",
       {same("auid", "aflow:0000000000000002"),
        same("compound", "Ta2"),
        {"species", q("Ta"), R"(["Ta"])"},
        {"composition", q("2"), "[2]"},
        {"volume_cell", q("36.12682"), "36.12682"},
        {"density", q("16.634236"), "16.634236"},
        {"geometry", q(ta_conv), "[" + ta_conv + "]"},
        {"positions_fractional", q("0,0,0;0.5,0.5,0.5"),
         "[[0,0,0],[0.5,0.5,0.5]]"}}},
      {"rock-salt NaCl, primitive cell, 2 species",
       "aflow:0000000000000003",
       {"Cl", "Na"},
       "aflow:0000000000000003 | Cl1Na1 | Volume: 44.85154 | Density: 2.163728",
       {same("auid", "aflow:0000000000000003"),
        same("compound", "Cl1Na1"),
        {"species", q("Cl,Na"), R"(["Cl","Na"])"},
        {"composition", q("1,1"), "[1,1]"},
        {"volume_cell", q("44.85154"), "44.85154"},
        {"density", q("2.163728"), "2.163728"},
        {"geometry", q(nacl_geom), "[" + nacl_geom + "]"},
        {"positions_fractional", q("0,0,0;0.5,0.5,0.5"),
         "[[0,0,0],[0.5,0.5,0.5]]"}}},
  };
}

bool contains(const std::vector<std::string> &keys, const std::string &key) {
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

// JSON text of the entry with every field in the form `form`, except the
// fields named in `flip`, which take the other form, and the field
// `bad_key`, whose value text is `bad_value`.
std::string entry_json(const Entry &e, Form form,
                       const std::vector<std::string> &flip = {},
                       const std::string &bad_key = "",
                       const std::string &bad_value = "") {
  std::string out = "{";
  for (size_t i = 0; i < e.fields.size(); ++i) {
    const Field &f = e.fields[i];
    const bool typed = (form == Form::Typed) != contains(flip, f.key);
    if (i) out += ",";
    out += q(f.key) + ":" +
           (f.key == bad_key ? bad_value : typed ? f.typed : f.str);
  }
  return out + "}";
}

// parseAflowJson's structure, or the message of what it threw.
struct Parsed {
  bool ok = false;
  std::string error;
  Structure st;
};

Parsed try_parse(const std::string &json) {
  Parsed p;
  try {
    AflowReader reader;
    reader.parseAflowJson(json);
    p.st = reader.getStructure();
    p.ok = true;
  } catch (const std::exception &ex) {
    p.error = ex.what();
  }
  return p;
}

// The fields parseAflowJson sets: the label, the cell and, per atom, the
// element and the position. The doubles are compared exactly: both forms
// hand the same decimal text to strtod, and the cell and the positions are
// then computed by the same code.
void check_same(const Structure &got, const Structure &want) {
  CHECK(got.label == want.label);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      INFO("cell(" << i << "," << k << ")");
      CHECK(got.cell(i, k) == want.cell(i, k));
    }
  CHECK(got.atoms.size() == want.atoms.size());
  if (got.atoms.size() != want.atoms.size()) return;
  for (size_t n = 0; n < got.atoms.size(); ++n) {
    INFO("atom " << n);
    CHECK(got.atoms[n].symbol == want.atoms[n].symbol);
    CHECK(got.atoms[n].Z == want.atoms[n].Z);
    for (int k = 0; k < 3; ++k)
      CHECK(got.atoms[n].position[k] == want.atoms[n].position[k]);
  }
}

// The all-string entry's structure, checked against the entry's expected
// atoms and label so that a comparison with it is not one between two
// empty structures.
Structure reference(const Entry &e) {
  const Parsed p = try_parse(entry_json(e, Form::String));
  INFO("string form: " << p.error);
  REQUIRE(p.ok);
  REQUIRE(p.st.atoms.size() == e.symbols.size());
  for (size_t n = 0; n < e.symbols.size(); ++n)
    CHECK(p.st.atoms[n].symbol == e.symbols[n]);
  CHECK(p.st.label == e.label);
  return p.st;
}

}  // namespace

// Pins: each hand-written entry (bcc Ta, primitive cell with 1 atom; bcc Ta,
// conventional cell with 2 atoms; rock-salt NaCl with 2 species) parses in
// the typed form to exactly the Structure of its string form: the same
// label '<auid> | <compound> | Volume: <volume_cell> | Density: <density>',
// with each number written as served ('Volume: 18.06341', not a rounding of
// it), the same cell bit for bit, and the same atoms (element and position)
// in the same order. JSON integers (the 60 and 90 degree angles, the 0
// coordinates, the counts) are accepted where a number is expected, and the
// keys the reader does not use are ignored. The string forms are checked
// against the expected atoms and labels first.
// Before the fix: each typed entry throws nlohmann::json::type_error 302
// '[json.exception.type_error.302] type must be string, but is array' at
// "composition", the first typed field the reader reads (with
// root.value("composition", "N/A")); the string forms already pass.
TEST_CASE("AFLOW reader: a typed entry gives the same structure as its "
          "string form",
          "[structure_readers][aflow][aflow_typed]") {
  PeriodicTable::initialize();
  for (const Entry &e : entries()) {
    INFO(e.name);
    const Structure want = reference(e);
    const Parsed got = try_parse(entry_json(e, Form::Typed));
    INFO("typed form: " << got.error);
    CHECK(got.ok);
    if (got.ok) check_same(got.st, want);
  }
}

// Pins: the reader takes each field in either form, whatever the form of
// the others. For each of the three entries and each of the six fields
// whose forms differ (species, composition, volume_cell, density, geometry,
// positions_fractional), the entry with that field typed and the others
// strings, and the entry with that field a string and the others typed,
// parse to exactly the Structure of the all-string entry (label included).
// Before the fix: all 36 entries throw nlohmann::json::type_error 302 at
// their first typed field in the reader's order (composition, species,
// volume_cell, density, geometry, positions_fractional): 'type must be
// string, but is array' for species, composition, geometry and
// positions_fractional, '... but is number' for volume_cell and density.
TEST_CASE("AFLOW reader: each field is read in either form, whatever the "
          "form of the others",
          "[structure_readers][aflow][aflow_typed]") {
  PeriodicTable::initialize();
  for (const Entry &e : entries()) {
    INFO(e.name);
    const Structure want = reference(e);
    for (const std::string &key : kTypedKeys) {
      for (const Form base : {Form::String, Form::Typed}) {
        const std::string json = entry_json(e, base, {key});
        INFO((base == Form::String ? "only " + key + " typed: "
                                   : "only " + key + " a string: ")
             << json);
        const Parsed got = try_parse(json);
        INFO("error: " << got.error);
        CHECK(got.ok);
        if (got.ok) check_same(got.st, want);
      }
    }
  }
}

// Pins: a field that is neither in its string form nor in its typed form,
// or whose typed form has the wrong length, is refused with a
// std::runtime_error whose message names the field and the entry's auid
// (only the field when the auid itself is bad). Each bad value is tried in
// the NaCl entry with the other fields strings and with them typed:
//   geometry: 5 or 7 numbers, a string element, an object;
//   positions_fractional: a site of 2 or of 4 numbers, a flat list of six
//     numbers, a string element;
//   species: 3 symbols for 2 counts, a number element;
//   composition: 1 count for 2 species, counts adding up to 3 for 2 sites,
//     a string element;
//   volume_cell an array, density an object, auid a number, compound an
//   array.
// The species and composition count checks are the existing consistency
// check ('... is inconsistent: 3 species, 2 composition counts adding up to
// 2, and 2 positions.'), reached now from the typed form.
// Before the fix: every one of the 34 entries throws
// nlohmann::json::type_error 302 ('type must be string, but is array',
// '... is number' or '... is object'), which is not a std::runtime_error
// and names no field: with the other fields strings it is thrown at the bad
// field, with them typed at "composition" (at "compound" when that is the
// bad field, since the reader reads it first).
TEST_CASE("AFLOW reader: a malformed field in either form is refused, "
          "naming the field",
          "[structure_readers][aflow][aflow_typed]") {
  PeriodicTable::initialize();
  const Entry nacl = entries()[2];
  struct Bad {
    std::string key, value;
    std::vector<std::string> needles;
  };
  const std::string g = "3.98808224589";
  const std::vector<Bad> bad = {
      {"geometry", "[" + g + "," + g + "," + g + ",60,60]", {"geometry"}},
      {"geometry", "[" + g + "," + g + "," + g + ",60,60,60,60]", {"geometry"}},
      {"geometry", "[" + g + "," + g + "," + g + ",60,60,\"60\"]",
       {"geometry"}},
      {"geometry", R"({"a":3.98808224589})", {"geometry"}},
      {"positions_fractional", "[[0,0,0],[0.5,0.5]]",
       {"positions_fractional"}},
      {"positions_fractional", "[[0,0,0],[0.5,0.5,0.5,0.5]]",
       {"positions_fractional"}},
      {"positions_fractional", "[0,0,0,0.5,0.5,0.5]",
       {"positions_fractional"}},
      {"positions_fractional", R"([[0,0,0],[0.5,"0.5",0.5]])",
       {"positions_fractional"}},
      {"species", R"(["Cl","Na","K"])", {"species", "composition"}},
      {"species", R"(["Cl",11])", {"species"}},
      {"composition", "[1]", {"species", "composition"}},
      {"composition", "[1,2]", {"composition"}},
      {"composition", R"([1,"1"])", {"composition"}},
      {"volume_cell", "[44.85154]", {"volume_cell"}},
      {"density", R"({"value":2.163728})", {"density"}},
      {"auid", "3", {"auid"}},
      {"compound", R"(["Cl","Na"])", {"compound"}},
  };
  for (const Bad &b : bad) {
    for (const Form base : {Form::String, Form::Typed}) {
      const std::string json = entry_json(nacl, base, {}, b.key, b.value);
      INFO((base == Form::String ? "others strings: " : "others typed: ")
           << json);
      AflowReader reader;
      CHECK_THROWS_AS(reader.parseAflowJson(json), std::runtime_error);
      std::vector<std::string> needles = b.needles;
      if (b.key != "auid") needles.push_back(nacl.auid);
      for (const std::string &needle : needles) {
        AflowReader r;
        CHECK_THROWS_WITH(r.parseAflowJson(json), Catch::Contains(needle));
      }
    }
  }
}
// Pins: a composition count that is not a whole number, or is negative, is
// refused in either form with a std::runtime_error naming "composition" and
// the entry's auid: "1.4,0.6" and [1.4,0.6] (they round to 1 and 1, the
// right total for the two sites), "-1,3" and [-1,3] (whose total, 2, also
// matches the two sites).
// Before the fix, with the other fields strings: "1.4,0.6" is read as 1,1
// and the entry is accepted as NaCl; "-1,3" passes the count check (the
// size_t total wraps round to 2) and then gives three Na atoms, the third
// read past the end of the two-site position list (undefined behaviour, in
// practice no exception); [1.4,0.6] and [-1,3] throw
// nlohmann::json::type_error 302 at "composition". With the other fields
// typed every case throws type_error 302 (at "composition", or at "species"
// when the bad value is a string). type_error is not a std::runtime_error
// and names no field.
TEST_CASE("AFLOW reader: a composition count that is not a whole number, "
          "or is negative, is refused",
          "[structure_readers][aflow][aflow_typed]") {
  PeriodicTable::initialize();
  const Entry nacl = entries()[2];
  const std::vector<std::string> bad = {
      R"("1.4,0.6")", "[1.4,0.6]", R"("-1,3")", "[-1,3]"};
  for (const std::string &value : bad) {
    for (const Form base : {Form::String, Form::Typed}) {
      const std::string json =
          entry_json(nacl, base, {}, "composition", value);
      INFO(json);
      AflowReader reader;
      CHECK_THROWS_AS(reader.parseAflowJson(json), std::runtime_error);
      const std::vector<std::string> needles = {"composition", nacl.auid};
      for (const std::string &needle : needles) {
        AflowReader r;
        CHECK_THROWS_WITH(r.parseAflowJson(json), Catch::Contains(needle));
      }
    }
  }
}

/* Pins: the string form is read strictly: in the NaCl entry (other fields
 * strings),
 *  - a geometry item that is not exactly one number ("3.98808224589x") is
 *    refused, naming "geometry" and the auid;
 *  - an empty item ("Cl,,Na" in species, "1,,1" in composition) is refused,
 *    naming the field;
 *  - blanks around an item are allowed: "Cl, Na" gives the same structure as
 *    "Cl,Na";
 *  - an entry without "species" (or "composition") is refused, naming it:
 *    has no "species";
 *  - text that is not JSON is a std::runtime_error saying it cannot be read
 *    as JSON.
 * Before the fix: "3.98808224589x" was read as 3.98808224589, empty items
 * were dropped (so "Cl,,Na" was read as Cl, Na), "Cl, Na" failed with
 * "Unknown symbol:  Na", a missing species gave "Unknown symbol: N/A", and
 * text that is not JSON threw nlohmann's parse_error. */
TEST_CASE("AFLOW reader: the string form is read strictly",
          "[structure_readers][aflow]") {
  PeriodicTable::initialize();
  const Entry nacl = entries()[2];
  struct Bad { std::string key, value, needle; };
  const std::vector<Bad> bad = {
      {"geometry", q("3.98808224589x,3.98808224589,3.98808224589,60,60,60"), "geometry"},
      {"species", q("Cl,,Na"), "species"},
      {"composition", q("1,,1"), "composition"}};
  for (const Bad &b : bad) {
    const std::string json = entry_json(nacl, Form::String, {}, b.key, b.value);
    INFO(json);
    AflowReader reader;
    CHECK_THROWS_AS(reader.parseAflowJson(json), std::runtime_error);
    AflowReader r;
    CHECK_THROWS_WITH(r.parseAflowJson(json),
                      Catch::Contains(b.needle) && Catch::Contains(nacl.auid));
  }
  {
    AflowReader plain, spaced;
    plain.parseAflowJson(entry_json(nacl, Form::String));
    REQUIRE_NOTHROW(spaced.parseAflowJson(
        entry_json(nacl, Form::String, {}, "species", q("Cl, Na"))));
    REQUIRE(spaced.getStructure().natoms() == plain.getStructure().natoms());
    for (size_t a = 0; a < plain.getStructure().natoms(); ++a)
      CHECK(spaced.getStructure().atoms[a].symbol == plain.getStructure().atoms[a].symbol);
  }
  for (const std::string missing : {"species", "composition"}) {
    Entry e = nacl;
    e.fields.erase(std::remove_if(e.fields.begin(), e.fields.end(),
                                  [&](const Field &f) { return f.key == missing; }),
                   e.fields.end());
    const std::string json = entry_json(e, Form::String);
    INFO(json);
    AflowReader r;
    CHECK_THROWS_WITH(r.parseAflowJson(json),
                      Catch::Contains("has no \"" + missing + "\""));
  }
  {
    AflowReader r;
    CHECK_THROWS_AS(r.parseAflowJson("{not json"), std::runtime_error);
    AflowReader r2;
    CHECK_THROWS_WITH(r2.parseAflowJson("{not json"), Catch::Contains("cannot be read as JSON"));
  }
}
