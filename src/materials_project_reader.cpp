#include <tadah/mlip/structure_readers/materials_project_reader.h>
#include <tadah/mlip/data_providers/fetch_failure.h>
#include <tadah/mlip/structure.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
namespace tadah {
namespace mlip {

MaterialsProjectReader::MaterialsProjectReader(const std::string &apiKey)
  : m_dataProvider(std::make_unique<MaterialsProjectDataProvider>(apiKey))
{
}

MaterialsProjectReader::MaterialsProjectReader(
    std::unique_ptr<IMaterialsDataProvider> provider)
  : m_dataProvider(std::move(provider)) {
  if (!m_dataProvider)
    throw std::invalid_argument(
        "internal error: the Materials Project reader has no data "
        "provider");
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
    throw std::runtime_error(failed_fetch_message("Materials Project",
                                                  "material", mpID, rr));
  // The structure replaces the previous one only when the material is read.
  m_structure = parseMpJson(rr.response, mpID);
}

// parseMpJson processes the JSON from the Materials Project
// "materials/core" endpoint. "data" is an array of results. The first
// result is a "MaterialsDoc" containing a "structure" sub-field. The
// "structure" sub-field is a JSON representation of a Pymatgen structure
// that includes a "lattice.matrix" and a "sites" list. Lattice vectors
// and atomic positions are returned in a Structure. Every error names the
// material, "Materials Project material '<mpID>': <what>", the JSON
// library's own errors included.
Structure MaterialsProjectReader::parseMpJson(const std::string &jsonContent,
                                              const std::string &mpID) {
  using json = nlohmann::json;
  auto fail = [&mpID](const std::string &what) {
    return std::runtime_error("Materials Project material '" + mpID +
                              "': " + what);
  };
  Structure result;
  try {
    json root = json::parse(jsonContent);

    if (!root.contains("data") || !root["data"].is_array() || root["data"].empty()) {
      throw fail("'data' is missing/empty.");
    }
    auto doc = root["data"][0];

    if (!doc.contains("structure")) {
      throw fail("no 'structure' field found.");
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
    result.label = label.str();

    // Extract the 3x3 matrix
    if (!structureJson.contains("lattice") || !structureJson["lattice"].contains("matrix")) {
      throw fail("No 'lattice.matrix' found.");
    }
    auto mat = structureJson["lattice"]["matrix"];
    if (!mat.is_array() || mat.size() != 3) {
      throw fail("'matrix' must be array of length 3.");
    }
    for (int i=0; i<3; i++){
      if (!mat[i].is_array() || mat[i].size() != 3) {
        throw fail("each row of 'matrix' must have length 3.");
      }
      result.cell(i,0) = mat[i][0].get<double>();
      result.cell(i,1) = mat[i][1].get<double>();
      result.cell(i,2) = mat[i][2].get<double>();
    }

    // parse the "sites" array
    if (!structureJson.contains("sites") || !structureJson["sites"].is_array()) {
      throw fail("'sites' is missing or not an array.");
    }
    auto sites = structureJson["sites"];

    size_t n = 0;
    for (auto &st : sites) {
      ++n;
      if (!st.contains("abc") || !st["abc"].is_array() || st["abc"].size() != 3) {
        throw fail("site " + std::to_string(n) + " is missing 'abc'.");
      }
      double fx = st["abc"][0].get<double>();
      double fy = st["abc"][1].get<double>();
      double fz = st["abc"][2].get<double>();

      // Fractional -> Cartesian
      double x = fx*result.cell(0,0) + fy*result.cell(1,0) + fz*result.cell(2,0);
      double y = fx*result.cell(0,1) + fy*result.cell(1,1) + fz*result.cell(2,1);
      double z = fx*result.cell(0,2) + fy*result.cell(1,2) + fz*result.cell(2,2);

      // tadah::core::Element from either "label" or "species[0].element"
      std::string elemSymbol;
      if (st.contains("label") && st["label"].is_string()) {
        elemSymbol = st["label"].get<std::string>();
      } else if (st.contains("species") && st["species"].is_array() && !st["species"].empty()) {
        elemSymbol = st["species"][0]["element"].get<std::string>();
      } else {
        throw fail("cannot determine the element symbol of site " +
                   std::to_string(n) + ".");
      }

      Atom a;
      try {
        static_cast<tadah::core::Element&>(a) =
            tadah::core::PeriodicTable::find_by_symbol(elemSymbol);
      } catch (const std::runtime_error &) {
        throw fail("site " + std::to_string(n) + " has element '" +
                   elemSymbol +
                   "', which is not an element symbol Tadah knows (H to Og).");
      }
      a.position[0] = x;
      a.position[1] = y;
      a.position[2] = z;
      result.atoms.push_back(a);
    }
  } catch (const json::exception &e) {
    throw fail(e.what());
  }
  return result;
}
} // namespace mlip
} // namespace tadah
