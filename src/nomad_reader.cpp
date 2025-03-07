#include <tadah/mlip/structure_readers/nomad_reader.h>
#include <tadah/mlip/structure.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <cmath>

NomadReader::NomadReader()
  : m_dataProvider(std::make_unique<NomadDataProvider>()) {
}

void NomadReader::read(const std::string &entryID) {
  fetchAndParseNomad(entryID);
}

Structure NomadReader::getStructure() const {
  return m_structure;
}

void NomadReader::fetchAndParseNomad(const std::string &entryID) {
  RawResponse rr = m_dataProvider->QueryByID(entryID);
  if (rr.httpStatus != 200) {
    std::ostringstream oss;
    oss << "NomadReader: HTTP " << rr.httpStatus
        << " fetching entry '" << entryID << "'.";
    throw std::runtime_error(oss.str());
  }
  parseNomadJson(rr.response);
}

/**
 * @brief parseNomadJson() extracts lattice vectors, site positions, etc.
 */
void NomadReader::parseNomadJson(const std::string &jsonContent) {
  using json = nlohmann::json;
  auto root = json::parse(jsonContent);

  if (!root.contains("data") || !root["data"].is_object()) {
    throw std::runtime_error("NomadReader: 'data' missing or not an object.");
  }
  auto dataObj = root["data"];

  if (!dataObj.contains("archive") || !dataObj["archive"].is_object()) {
    throw std::runtime_error("NomadReader: 'archive' missing or not object.");
  }
  auto archive = dataObj["archive"];

  // Additional checks, read synergy, etc. 
  // ... same parsing logic as before ...

  // Minimal illustration to show that we do the normal JSON parse
  // to fill m_structure. For brevity, not showing all lines from the original code.
  if (!archive.contains("results") || !archive["results"].contains("material")) {
    throw std::runtime_error("NomadReader: no results.material found.");
  }

  auto material = archive["results"]["material"];
  std::string spgSymbol = material.value("symmetry", json())
                                  .value("space_group_symbol", "???");

  auto topologies = material["topology"];
  if (!topologies.is_array() || topologies.empty()) {
    throw std::runtime_error("NomadReader: no topology array found.");
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
  m_structure.label = labelStream.str();

  // Transfer lattice vectors
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m_structure.cell(i, j) = lattVecs[i][j].get<double>();
    }
  }

  // Retrieve cartesian positions and species
  if (!optimade.contains("cartesian_site_positions") ||
      !optimade["cartesian_site_positions"].is_array()) {
    throw std::runtime_error("parseNomadJson: 'cartesian_site_positions' missing/not array.");
  }
  auto cartesianSites = optimade["cartesian_site_positions"];

  if (!optimade.contains("species_at_sites") ||
      !optimade["species_at_sites"].is_array()) {
    throw std::runtime_error("parseNomadJson: 'species_at_sites' missing/not array.");
  }
  auto speciesSites = optimade["species_at_sites"];

  if (cartesianSites.size() != speciesSites.size()) {
    throw std::runtime_error("parseNomadJson: Position-species array size mismatch.");
  }

  m_structure.atoms.reserve(cartesianSites.size());

  // Combine site position with species
  for (size_t i = 0; i < cartesianSites.size(); ++i) {
    auto &pos = cartesianSites[i];
    if (!pos.is_array() || pos.size() != 3) {
      throw std::runtime_error("parseNomadJson: Site position requires 3 coordinates.");
    }
    double x = pos[0].get<double>();
    double y = pos[1].get<double>();
    double z = pos[2].get<double>();

    Atom atom;
    Element e = PeriodicTable().find_by_symbol(speciesSites[i].get<std::string>());
    static_cast<Element &>(atom) = e;
    atom.position[0] = x;
    atom.position[1] = y;
    atom.position[2] = z;

    m_structure.atoms.push_back(atom);
  }
  // Done parsing
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
