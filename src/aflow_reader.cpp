#include <tadah/mlip/structure_readers/aflow_reader.h>
#include <tadah/core/number_format.h>
#include <nlohmann/json.hpp>
#include <array>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <vector>
namespace tadah {
namespace mlip {

namespace {

using json = nlohmann::json;

/*
  An aflowlib entry comes in two encodings, and parseAflowJson reads each
  field in either, whatever the encoding of the others:

    field                 string form                typed form
    geometry              "a,b,c,alpha,beta,gamma"   [a,b,c,alpha,beta,gamma]
    species               "Cl,Na"                    ["Cl","Na"]
    composition           "1,1"                      [1,1]
    positions_fractional  "0,0,0;0.5,0.5,0.5"        [[0,0,0],[0.5,0.5,0.5]]
    volume_cell, density  "44.85154"                 44.85154
    compound, auid        "Cl1Na1"                   (a string in both)

  The string form is that of the AFLOW REST API's text output; the typed
  form is that of the entry file aflowlib.json, which AFLOWDataProvider
  fetches. A field in neither form is refused, naming the entry and the
  field.
*/

// Refuses entry `auid`: "AflowReader: entry '<auid>': <what>".
[[noreturn]] void refuse(const std::string &auid, const std::string &what) {
  throw std::runtime_error("AflowReader: entry '" + auid + "': " + what);
}

// "not a JSON array." and the like, for a value in neither form.
std::string not_a(const json &v) {
  return std::string("not a JSON ") + v.type_name() + ".";
}

// The value of `key`, or nullptr when the entry has no such key.
const json *field(const json &root, const char *key) {
  const auto it = root.find(key);
  return it == root.end() ? nullptr : &*it;
}

// `text` without the blanks (spaces and tabs) around it.
std::string trimmed(const std::string &text) {
  const std::size_t b = text.find_first_not_of(" \t");
  if (b == std::string::npos) return "";
  const std::size_t e = text.find_last_not_of(" \t");
  return text.substr(b, e - b + 1);
}

// The comma-separated items of `text`, each trimmed; none when `text` is
// blank. An empty item ("1,,2", "1,2,") is kept, for the caller to refuse.
std::vector<std::string> comma_items(const std::string &text) {
  std::vector<std::string> items;
  if (trimmed(text).empty()) return items;
  std::size_t from = 0;
  while (true) {
    const std::size_t comma = text.find(',', from);
    items.push_back(trimmed(text.substr(from, comma - from)));
    if (comma == std::string::npos) break;
    from = comma + 1;
  }
  return items;
}

// Whether `item` is one finite decimal number and nothing else (in `out`).
bool text_number(const std::string &item, double &out) {
  return tadah::core::parse_decimal_finite(item, out);
}

// Whether `v` is a JSON number (in `out`); an integer is one. It is
// finite: json::parse refuses a literal that overflows.
bool json_number(const json &v, double &out) {
  if (!v.is_number()) return false;
  out = v.get<double>();
  return true;
}

// "compound" (and "auid"): a string in both forms.
std::string string_field(const json &root, const char *key,
                         const std::string &absent, const std::string &auid) {
  const json *v = field(root, key);
  if (!v) return absent;
  if (!v->is_string()) {
    refuse(auid, "\"" + std::string(key) + "\" must be a string, " +
                     not_a(*v));
  }
  return v->get<std::string>();
}

// "volume_cell" and "density", which only go into the label: the string as
// served, or the number as JSON writes it (j.dump(), "18.06341"), so that
// both forms give the same label.
std::string label_number(const json &root, const char *key,
                         const std::string &auid) {
  const json *v = field(root, key);
  if (!v) return "N/A";
  if (v->is_string()) return v->get<std::string>();
  double x;
  if (json_number(*v, x)) return v->dump();
  refuse(auid, "\"" + std::string(key) + "\" must be a number or a string, " +
                   not_a(*v));
}

// "geometry": a, b, c, alpha, beta, gamma.
std::array<double, 6> read_geometry(const json &v, const std::string &auid) {
  const std::string need =
      "\"geometry\" needs six numbers (a,b,c,alpha,beta,gamma), got ";
  std::vector<double> vals;
  if (v.is_string()) {
    const std::string text = v.get<std::string>();
    for (const std::string &item : comma_items(text)) {
      double x;
      if (!text_number(item, x)) refuse(auid, need + "'" + text + "'.");
      vals.push_back(x);
    }
    if (vals.size() != 6) refuse(auid, need + "'" + text + "'.");
  } else if (v.is_array()) {
    for (const json &e : v) {
      double x;
      if (!json_number(e, x)) refuse(auid, need + v.dump() + ".");
      vals.push_back(x);
    }
    if (vals.size() != 6) refuse(auid, need + v.dump() + ".");
  } else {
    refuse(auid, "\"geometry\" must be a string \"a,b,c,alpha,beta,gamma\" "
                 "or an array of six numbers, " + not_a(v));
  }
  return {vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]};
}

// "species": the element symbols, in the order of "composition".
std::vector<std::string> read_species(const json &v, const std::string &auid) {
  std::vector<std::string> out;
  if (v.is_string()) {
    const std::string text = v.get<std::string>();
    for (const std::string &item : comma_items(text)) {
      if (item.empty())
        refuse(auid, "\"species\" has an empty symbol in '" + text + "'.");
      out.push_back(item);
    }
  } else if (v.is_array()) {
    for (const json &e : v) {
      if (!e.is_string() || e.get<std::string>().empty()) {
        refuse(auid, "\"species\" must hold element symbols (strings), got " +
                         e.dump() + " in " + v.dump() + ".");
      }
      out.push_back(e.get<std::string>());
    }
  } else {
    refuse(auid, "\"species\" must be a string \"Cl,Na,...\" or an array of "
                 "element symbols, " + not_a(v));
  }
  return out;
}

// One "composition" count, which must be a whole number of atoms, 0 or
// more (a count of 1.4 is not rounded, and a negative one is not wrapped).
double composition_count(double x, const std::string &shown,
                         const std::string &whole, const std::string &auid) {
  if (!(x >= 0.0) || x != std::floor(x)) {
    refuse(auid, "a \"composition\" count must be a whole number of atoms, "
                 "0 or more, got " + shown + " in " + whole + ".");
  }
  return x;
}

// "composition": the number of atoms of each species. The counts are held
// as doubles, whole and not negative; parseAflowJson converts them once
// their total is known to be the number of positions.
std::vector<double> read_composition(const json &v, const std::string &auid) {
  std::vector<double> out;
  if (v.is_string()) {
    const std::string text = v.get<std::string>();
    for (const std::string &item : comma_items(text)) {
      double x;
      if (!text_number(item, x)) {
        refuse(auid, "'" + item + "' in \"composition\" '" + text +
                         "' is not a number.");
      }
      out.push_back(composition_count(x, item, "'" + text + "'", auid));
    }
  } else if (v.is_array()) {
    for (const json &e : v) {
      double x;
      if (!json_number(e, x)) {
        refuse(auid, "\"composition\" must hold numbers, got " + e.dump() +
                         " in " + v.dump() + ".");
      }
      out.push_back(composition_count(x, e.dump(), v.dump(), auid));
    }
  } else {
    refuse(auid, "\"composition\" must be a string \"1,1,...\" or an array "
                 "of numbers, " + not_a(v));
  }
  return out;
}

// "positions_fractional": three fractional coordinates per site.
std::vector<std::array<double, 3>> read_positions(const json &v,
                                                  const std::string &auid) {
  std::vector<std::array<double, 3>> out;
  if (v.is_string()) {
    // "f1,f2,f3;f1,f2,f3;...": an empty site (a trailing ';') is skipped.
    std::stringstream sPos(v.get<std::string>());
    std::string triplet;
    while (std::getline(sPos, triplet, ';')) {
      if (trimmed(triplet).empty()) continue;
      const std::vector<std::string> fields = comma_items(triplet);
      if (fields.size() != 3) {
        refuse(auid, "a \"positions_fractional\" site needs three numbers, "
                     "got '" + triplet + "'.");
      }
      std::array<double, 3> f{0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        if (!text_number(fields[i], f[i])) {
          refuse(auid, "'" + fields[i] + "' in \"positions_fractional\" is "
                       "not a number.");
        }
      }
      out.push_back(f);
    }
  } else if (v.is_array()) {
    for (std::size_t n = 0; n < v.size(); ++n) {
      const json &site = v[n];
      std::array<double, 3> f{0, 0, 0};
      bool ok = site.is_array() && site.size() == 3;
      for (int i = 0; ok && i < 3; ++i) ok = json_number(site[i], f[i]);
      if (!ok) {
        refuse(auid, "site " + std::to_string(n + 1) +
                         " of \"positions_fractional\" needs three numbers, "
                         "got " + site.dump() + ".");
      }
      out.push_back(f);
    }
  } else {
    refuse(auid, "\"positions_fractional\" must be a string "
                 "\"f1,f2,f3;f1,f2,f3;...\" or an array of sites of three "
                 "numbers, " + not_a(v));
  }
  return out;
}

}  // namespace

AflowReader::AflowReader()
  : m_dataProvider(std::make_unique<AFLOWDataProvider>()) {
}

void AflowReader::read(const std::string &auid) {
  fetchAndParseAflowByAuid(auid);
}

Structure AflowReader::getStructure() const {
  return m_structure;
}

void AflowReader::fetchAndParseAflowByAuid(const std::string &auid) {
  RawResponse rr = m_dataProvider->QueryByID(auid);
  if (rr.httpStatus != 200) {
    std::ostringstream oss;
    oss << "AflowReader: HTTP " << rr.httpStatus
        << " for AUID '" << auid << "'.";
    throw std::runtime_error(oss.str());
  }
  parseAflowJson(rr.response);
}
/*
  parseAflowJson(jsonContent):
    Interprets an aflowlib entry (a JSON object), which includes
    lattice parameters ("geometry"), composition, species list,
    and fractional positions. Each field is read in its string form or in
    its typed form (the table at the top of this file).

    - "auid", "compound", "volume_cell" and "density" make the label
      '<auid> | <compound> | Volume: <volume_cell> | Density: <density>',
      each number written as served (a typed one as JSON writes it).
    - "geometry" is used to build a 3×3 cell matrix via makeCellMatrix()
      (a along x, b in the xy plane; row i is lattice vector a_i).
    - "positions_fractional" (relative to "geometry") plus composition &
      species become the final atoms, at r = f0 a0 + f1 a1 + f2 a2.
      "positions_cartesian" is not used: it is expressed in AFLOW's own
      frame, which in general is not the orientation built here.
    - An entry without "geometry", "species", "composition" or
      "positions_fractional" is refused.

  These are refused with a std::runtime_error naming the entry's auid and
  the field: a field in neither form, a geometry of other than six numbers,
  a site of other than three, a composition count that is not a whole
  number 0 or more, and an entry whose species, composition counts and
  positions do not match. On any refusal the reader's structure is left as
  it was.
*/
void AflowReader::parseAflowJson(const std::string &jsonContent) {
  json root;
  try {
    root = json::parse(jsonContent);
  } catch (const json::exception &e) {
    throw std::runtime_error(std::string("AflowReader: the entry cannot be "
                                         "read as JSON: ") + e.what());
  }
  if (!root.is_object()) {
    throw std::runtime_error("AflowReader: an aflowlib entry is a JSON "
                             "object, " + not_a(root));
  }
  Structure st;

  //------------------------------------------------------------------
  // 1) Build a label with: AUID, compound, volume, density. The auid is
  //    read first: every later message names the entry by it.
  //------------------------------------------------------------------
  std::string auid = "N/A";
  if (const json *v = field(root, "auid")) {
    if (!v->is_string()) {
      throw std::runtime_error("AflowReader: \"auid\" must be a string, " +
                               not_a(*v));
    }
    auid = v->get<std::string>();
  }
  const std::string compound =
      string_field(root, "compound", "UnknownCompound", auid);
  const std::string volumeCell = label_number(root, "volume_cell", auid);
  const std::string density    = label_number(root, "density", auid);

  std::ostringstream lbl;
  lbl << auid << " | " << compound
      << " | Volume: "     << volumeCell
      << " | Density: "    << density;
  st.label = lbl.str();

  //------------------------------------------------------------------
  // 2) Geometry (a, b, c, alpha, beta, gamma) to a 3×3 lattice.
  //------------------------------------------------------------------
  const json *geometry = field(root, "geometry");
  if (!geometry) {
    throw std::runtime_error("AflowReader: entry '" + auid +
                             "' has no \"geometry\" (a,b,c,alpha,beta,gamma).");
  }
  const std::array<double, 6> g = read_geometry(*geometry, auid);
  double cellMat[3][3];
  makeCellMatrix(g[0], g[1], g[2], g[3], g[4], g[5], cellMat);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      st.cell(i, j) = cellMat[i][j];
    }
  }

  //------------------------------------------------------------------
  // 3) Species (e.g. "Ag,K,Se,Ta") and composition, the number of atoms
  //    of each (e.g. "6,6,16,4").
  //------------------------------------------------------------------
  const json *species = field(root, "species");
  if (!species) {
    throw std::runtime_error("AflowReader: entry '" + auid +
                             "' has no \"species\".");
  }
  const std::vector<std::string> speciesList = read_species(*species, auid);

  const json *composition = field(root, "composition");
  if (!composition) {
    throw std::runtime_error("AflowReader: entry '" + auid +
                             "' has no \"composition\".");
  }
  const std::vector<double> stoichCount = read_composition(*composition, auid);

  //------------------------------------------------------------------
  // 4) Fractional positions, each converted with the rows of the cell:
  //    r = f0*a0 + f1*a1 + f2*a2.
  //------------------------------------------------------------------
  const json *fractional = field(root, "positions_fractional");
  if (!fractional) {
    throw std::runtime_error("AflowReader: entry '" + auid +
                             "' has no \"positions_fractional\".");
  }
  std::vector<std::array<double,3>> positions;
  for (const std::array<double,3> &f : read_positions(*fractional, auid)) {
    std::array<double,3> r{0,0,0};
    for (int k = 0; k < 3; ++k) {
      r[k] = f[0] * cellMat[0][k] + f[1] * cellMat[1][k] +
             f[2] * cellMat[2][k];
    }
    positions.push_back(r);
  }

  //------------------------------------------------------------------
  // 5) Construct the final atoms: one species per composition count, and
  //    as many sites as the counts add up to. The counts are whole and not
  //    negative, so their floating-point sum is at least each count: a
  //    total equal to the number of positions bounds every count by it,
  //    and the sum and the counts are then exact.
  //------------------------------------------------------------------
  double totalCount = 0.0;
  for (double c : stoichCount) totalCount += c;

  if (speciesList.empty() || speciesList.size() != stoichCount.size() ||
      totalCount != static_cast<double>(positions.size())) {
    throw std::runtime_error(
      "AflowReader: entry '" + auid + "' is inconsistent: " +
      std::to_string(speciesList.size()) + " species, " +
      std::to_string(stoichCount.size()) + " composition counts adding up to " +
      tadah::core::format_roundtrip(totalCount) + ", and " +
      std::to_string(positions.size()) + " positions.");
  }

  // Distribute species among positions in order.
  st.atoms.reserve(positions.size());
  size_t idxPos = 0;
  for (size_t iSpec = 0; iSpec < speciesList.size(); ++iSpec) {
    tadah::core::Element e = tadah::core::PeriodicTable().find_by_symbol(speciesList[iSpec]);
    const size_t count = static_cast<size_t>(stoichCount[iSpec]);
    for (size_t rep = 0; rep < count; ++rep) {
      Atom atom;
      static_cast<tadah::core::Element &>(atom) = e;
      atom.position[0] = positions[idxPos][0];
      atom.position[1] = positions[idxPos][1];
      atom.position[2] = positions[idxPos][2];
      st.atoms.push_back(atom);
      ++idxPos;
    }
  }
  m_structure = std::move(st);
}

/*
  makeCellMatrix():
    Builds the standard 3×3 lattice vectors from:
       - a,b,c (lattice lengths)
       - alpha, beta, gamma (lattice angles in degrees)

    The formula is standard:
      R0 = ( a, 0, 0 )
      R1 = ( b cos(γ),  b sin(γ),         0 )
      R2 = ( c cos(β),  c [cos(α) − cos(β) cos(γ)] / sin(γ),  
            c sqrt( 1 − cos²(β) − ... ) )

    The final line for c is chosen so the angles are satisfied.
*/
void AflowReader::makeCellMatrix(double a, double b, double c,
                                 double alphaDeg, double betaDeg, double gammaDeg,
                                 double (&cell)[3][3]) {
  double alpha = alphaDeg * M_PI / 180.0;
  double beta  = betaDeg  * M_PI / 180.0;
  double gamma = gammaDeg * M_PI / 180.0;

  cell[0][0] = a;
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;

  cell[1][0] = b * std::cos(gamma);
  cell[1][1] = b * std::sin(gamma);
  cell[1][2] = 0.0;

  cell[2][0] = c * std::cos(beta);

  cell[2][1] = c * (std::cos(alpha) - std::cos(beta)*std::cos(gamma)) 
                 / std::sin(gamma);

  double term = 1.0
                - std::cos(beta) * std::cos(beta)
                - std::pow(
                    (std::cos(alpha) 
                       - std::cos(beta)*std::cos(gamma)) 
                    / std::sin(gamma),
                    2.0
                  );
  // term <= 0: no cell has these lengths and angles (e.g. gamma > alpha +
  // beta), or the cell is flat.
  if (!(a > 0.0 && b > 0.0 && c > 0.0) || !(term > 0.0)) {
    std::ostringstream msg;
    msg << "AflowReader: geometry " << a << "," << b << "," << c << ","
        << alphaDeg << "," << betaDeg << "," << gammaDeg
        << " (a,b,c,alpha,beta,gamma) describes no cell.";
    throw std::runtime_error(msg.str());
  }
  cell[2][2] = c * std::sqrt(term);
}
} // namespace mlip
} // namespace tadah
