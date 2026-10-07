#include <tadah/mlip/structure_readers/nomad_reader.h>
#include <tadah/mlip/data_providers/fetch_failure.h>
#include <tadah/mlip/structure.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <cmath>
namespace tadah {
namespace mlip {

NomadReader::NomadReader()
  : m_dataProvider(std::make_unique<NomadDataProvider>()) {
}

NomadReader::NomadReader(std::unique_ptr<IMaterialsDataProvider> provider)
  : m_dataProvider(std::move(provider)) {
  if (!m_dataProvider)
    throw std::invalid_argument("NomadReader: the data provider is null.");
}

void NomadReader::read(const std::string &entryID) {
  fetchAndParseNomad(entryID);
}

Structure NomadReader::getStructure() const {
  return m_structure;
}

void NomadReader::fetchAndParseNomad(const std::string &entryID) {
  RawResponse rr = m_dataProvider->QueryByID(entryID);
  if (rr.httpStatus != 200)
    throw std::runtime_error(
        failed_fetch_message("NomadReader", "entry", entryID, rr));
  // The structure replaces the previous one only when the entry is read.
  m_structure = parseNomadJson(rr.response, entryID);
}

/**
 * @brief parseNomadJson() extracts lattice vectors, site positions, etc.
 *
 * Every error names the entry: "NomadReader: entry '<entryID>': <what>",
 * the JSON library's own errors (text that is not JSON, a value of another
 * type) included.
 */
Structure NomadReader::parseNomadJson(const std::string &jsonContent,
                                      const std::string &entryID) {
  using json = nlohmann::json;
  auto fail = [&entryID](const std::string &what) {
    return std::runtime_error("NomadReader: entry '" + entryID + "': " + what);
  };
  Structure st;
  try {
    auto root = json::parse(jsonContent);

    if (!root.contains("data") || !root["data"].is_object()) {
      throw fail("'data' missing or not an object.");
    }
    auto dataObj = root["data"];

    if (!dataObj.contains("archive") || !dataObj["archive"].is_object()) {
      throw fail("'archive' missing or not object.");
    }
    auto archive = dataObj["archive"];

    if (!archive.contains("results") || !archive["results"].contains("material")) {
      throw fail("no results.material found.");
    }

    auto material = archive["results"]["material"];
    std::string spgSymbol = material.value("symmetry", json())
                                    .value("space_group_symbol", "???");

    auto topologies = material["topology"];
    if (!topologies.is_array() || topologies.empty()) {
      throw fail("no topology array found.");
    }
    auto cellData = topologies[0]["cell"];
    double volume = cellData.value("volume", 0.0);
    double density = cellData.value("mass_density", 0.0);

    auto optimade = archive["metadata"].value("optimade", json());
    auto lattVecs = optimade["lattice_vectors"];

    std::ostringstream labelStream;
    labelStream << "NOMAD Entry: " << dataObj.value("entry_id", "???")
                << " | SpaceGroup: " << spgSymbol
                << " | Vol: " << volume
                << " | Density: " << density;
    st.label = labelStream.str();

    // Transfer lattice vectors
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        st.cell(i, j) = lattVecs[i][j].get<double>();
      }
    }

    // Retrieve cartesian positions and species
    if (!optimade.contains("cartesian_site_positions") ||
        !optimade["cartesian_site_positions"].is_array()) {
      throw fail("'cartesian_site_positions' missing/not array.");
    }
    auto cartesianSites = optimade["cartesian_site_positions"];

    if (!optimade.contains("species_at_sites") ||
        !optimade["species_at_sites"].is_array()) {
      throw fail("'species_at_sites' missing/not array.");
    }
    auto speciesSites = optimade["species_at_sites"];

    if (cartesianSites.size() != speciesSites.size()) {
      throw fail("Position-species array size mismatch.");
    }

    st.atoms.reserve(cartesianSites.size());

    // Combine site position with species
    for (size_t i = 0; i < cartesianSites.size(); ++i) {
      auto &pos = cartesianSites[i];
      if (!pos.is_array() || pos.size() != 3) {
        throw fail("Site position requires 3 coordinates.");
      }
      double x = pos[0].get<double>();
      double y = pos[1].get<double>();
      double z = pos[2].get<double>();

      const std::string species = speciesSites[i].get<std::string>();
      Atom atom;
      try {
        static_cast<tadah::core::Element &>(atom) =
            tadah::core::PeriodicTable::find_by_symbol(species);
      } catch (const std::runtime_error &) {
        throw fail("site " + std::to_string(i + 1) + " has species '" +
                   species +
                   "', which is not an element symbol Tadah knows (H to Og).");
      }
      atom.position[0] = x;
      atom.position[1] = y;
      atom.position[2] = z;

      st.atoms.push_back(atom);
    }
  } catch (const json::exception &e) {
    throw fail(e.what());
  }
  return st;
}

void NomadReader::makeCellMatrix(double a, double b, double c,
                                 double alpha, double beta, double gamma,
                                 double (&cell)[3][3]) {
  // Same geometry math as before
  cell[0][0] = a;     cell[0][1] = 0.0;       cell[0][2] = 0.0;
  cell[1][0] = b * std::cos(gamma);
  cell[1][1] = b * std::sin(gamma);
  cell[1][2] = 0.0;
  cell[2][0] = c * std::cos(beta);
  cell[2][1] = c * (std::cos(alpha) - std::cos(beta)*std::cos(gamma)) / std::sin(gamma);

  double term = 1.0
    - std::cos(beta)*std::cos(beta)
    - std::pow((std::cos(alpha) - std::cos(beta)*std::cos(gamma)) / std::sin(gamma),2.0);

  cell[2][2] = c * std::sqrt(term > 0.0 ? term : 0.0);
}
} // namespace mlip
} // namespace tadah
