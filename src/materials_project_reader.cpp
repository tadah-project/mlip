#include <tadah/mlip/structure_readers/materials_project_reader.h>
#include <tadah/mlip/data_providers/fetch_failure.h>
#include <tadah/mlip/structure.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
namespace tadah {
namespace mlip {

static tadah::core::Element parse_element_mp(const std::string &elemName) {
  // Same helper from your original code
  return tadah::core::PeriodicTable().find_by_symbol(elemName);
}

MaterialsProjectReader::MaterialsProjectReader(const std::string &apiKey)
  : m_dataProvider(std::make_unique<MaterialsProjectDataProvider>(apiKey))
{
}

MaterialsProjectReader::MaterialsProjectReader(
    std::unique_ptr<IMaterialsDataProvider> provider)
  : m_dataProvider(std::move(provider)) {
  if (!m_dataProvider)
    throw std::invalid_argument(
        "MaterialsProjectReader: the data provider is null.");
}

void MaterialsProjectReader::read(const std::string &mpID) {
  fetchAndParseMP(mpID);
}

Structure MaterialsProjectReader::getStructure() const {
  return m_structure;
}

void MaterialsProjectReader::fetchAndParseMP(const std::string &mpID) {
  RawResponse rr = m_dataProvider->QueryByID(mpID);
  if (rr.httpStatus != 200)
    throw std::runtime_error(failed_fetch_message("MaterialsProjectReader",
                                                  "material", mpID, rr));
  parseMpJson(rr.response);
}

// parseMpJson processes the JSON from the Materials Project
// "materials/core" endpoint. "data" is an array of results. The first
// result is a "MaterialsDoc" containing a "structure" sub-field. The
// "structure" sub-field is a JSON representation of a Pymatgen structure
// that includes a "lattice.matrix" and a "sites" list. Lattice vectors
// and atomic positions are then stored in m_structure.
void MaterialsProjectReader::parseMpJson(const std::string &jsonContent) {
  using json = nlohmann::json;
  json root = json::parse(jsonContent);

  if (!root.contains("data") || !root["data"].is_array() || root["data"].empty()) {
    throw std::runtime_error("MaterialsProjectReader::parseMpJson - 'data' is missing/empty.");
  }
  auto doc = root["data"][0];

  if (!doc.contains("structure")) {
    throw std::runtime_error("MaterialsProjectReader - no 'structure' field found.");
  }
  auto structureJson = doc["structure"];

  std::string spg = to_string(doc["symmetry"]["symbol"]);
  std::string matID = to_string(doc["material_id"]);
  std::string vol = to_string(doc["volume"]);
  std::string dens = to_string(doc["density"]);
  std::string elems = to_string(doc["elements"]);

  std::ostringstream label;
  label << "MaterialsProject ID: " << matID
        << " | Symmetry: " << spg
        << " | Volume: " << vol
        << " | Density: " << dens
        << " | Elements: " << elems << " | ";
  m_structure.label = label.str();

  // Extract the 3x3 matrix
  if (!structureJson.contains("lattice") || !structureJson["lattice"].contains("matrix")) {
    throw std::runtime_error("parseMpJson: No 'lattice.matrix' found.");
  }
  auto mat = structureJson["lattice"]["matrix"];
  if (!mat.is_array() || mat.size() != 3) {
    throw std::runtime_error("parseMpJson: 'matrix' must be array of length 3.");
  }
  for (int i=0; i<3; i++){
    if (!mat[i].is_array() || mat[i].size() != 3) {
      throw std::runtime_error("parseMpJson: each row of 'matrix' must have length 3.");
    }
    m_structure.cell(i,0) = mat[i][0].get<double>();
    m_structure.cell(i,1) = mat[i][1].get<double>();
    m_structure.cell(i,2) = mat[i][2].get<double>();
  }

  // parse the "sites" array
  if (!structureJson.contains("sites") || !structureJson["sites"].is_array()) {
    throw std::runtime_error("parseMpJson: 'sites' is missing or not an array.");
  }
  auto sites = structureJson["sites"];

  for (auto &st : sites) {
    if (!st.contains("abc") || !st["abc"].is_array() || st["abc"].size() != 3) {
      throw std::runtime_error("parseMpJson: site is missing 'abc'.");
    }
    double fx = st["abc"][0].get<double>();
    double fy = st["abc"][1].get<double>();
    double fz = st["abc"][2].get<double>();

    // Fractional -> Cartesian
    double x = fx*m_structure.cell(0,0) + fy*m_structure.cell(1,0) + fz*m_structure.cell(2,0);
    double y = fx*m_structure.cell(0,1) + fy*m_structure.cell(1,1) + fz*m_structure.cell(2,1);
    double z = fx*m_structure.cell(0,2) + fy*m_structure.cell(1,2) + fz*m_structure.cell(2,2);

    // tadah::core::Element from either "label" or "species[0].element"
    std::string elemSymbol;
    if (st.contains("label") && st["label"].is_string()) {
      elemSymbol = st["label"].get<std::string>();
    } else if (st.contains("species") && st["species"].is_array() && !st["species"].empty()) {
      elemSymbol = st["species"][0]["element"].get<std::string>();
    } else {
      throw std::runtime_error("parseMpJson: cannot determine element symbol for a site.");
    }

    Atom a;
    tadah::core::Element e = parse_element_mp(elemSymbol);
    static_cast<tadah::core::Element&>(a) = e;
    a.position[0] = x;
    a.position[1] = y;
    a.position[2] = z;
    m_structure.atoms.push_back(a);
  }
}
} // namespace mlip
} // namespace tadah
