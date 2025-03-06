#include <tadah/mlip/structure_readers/nomad_reader.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <cmath>

// Collects HTTP response data into a std::string
static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* response = static_cast<std::string*>(userdata);
  size_t totalBytes = size * nmemb;
  response->append(ptr, totalBytes);
  return totalBytes;
}

// Default constructor. No special API token needed.
NomadReader::NomadReader() {
}

// Reads the structure by calling an internal fetchAndParseNomad for a given entryID.
void NomadReader::read(const std::string &entryID) {
  fetchAndParseNomad(entryID);
}

// Returns the last retrieved structure.
Structure NomadReader::getStructure() const {
  return m_structure;
}

/**
  fetchAndParseNomad() performs a POST request:
    https://nomad-lab.eu/prod/v1/api/v1/entries/<entryID>/archive/query
  The JSON body requests specific fields (cell, positions, species, etc.).
  Then parseNomadJson() processes the response.
*/
void NomadReader::fetchAndParseNomad(const std::string &entryID) {
  const std::string baseURL = "https://nomad-lab.eu/prod/v1/api/v1";
  const std::string url = baseURL + "/entries/" + entryID + "/archive/query";

  // JSON payload requesting relevant fields
  static const std::string requestBody = R"JSON(
  {
    "required": {
      "results": {
        "material": {
          "symmetry": {
            "space_group_symbol": "*"
          },
          "topology": {
            "cell": {
              "volume": "*",
              "mass_density": "*"
            }
          }
        }
      },
      "metadata": {
        "optimade": {
          "elements": "*",
          "chemical_formula_reduced": "*",
          "lattice_vectors": "*",
          "cartesian_site_positions": "*",
          "species_at_sites": "*"
        }
      }
    }
  }
  )JSON";

  // POST request with the JSON payload
  std::string response = httpPost(url, requestBody);

  parseNomadJson(response);
}

/**
  httpPost() sets JSON headers and payload for a POST request to @p url.
  The full response body is returned as a string.
*/
std::string NomadReader::httpPost(const std::string &url, const std::string &jsonBody) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("NomadReader::httpPost: Failed to init cURL.");
  }

  std::string response;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "accept: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonBody.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(jsonBody.size()));

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  // Optional user agent
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "NomadReader/1.0");
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << "[DEBUG] cURL error code: " << res << std::endl;
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    throw std::runtime_error("NomadReader::httpPost: " + std::string(curl_easy_strerror(res)));
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return response;
}

/**
  parseNomadJson() reads:
   - space group symbol from material.symmetry.space_group_symbol
   - label info (e.g. volume from topology[0].cell.volume)
   - a 3×3 lattice from metadata.optimade.lattice_vectors
   - positions and species from cartesian_site_positions and species_at_sites
*/
void NomadReader::parseNomadJson(const std::string &jsonContent) {
  using json = nlohmann::json;
  auto root = json::parse(jsonContent);

  // Basic top-level checks
  if (!root.is_object() || !root.contains("data")) {
    throw std::runtime_error("parseNomadJson: 'data' missing from root.");
  }
  auto dataObj = root["data"];
  if (!dataObj.contains("archive")) {
    throw std::runtime_error("parseNomadJson: 'data.archive' missing.");
  }
  auto archive = dataObj["archive"];

  // Check for results.material
  if (!archive.contains("results") || !archive["results"].contains("material")) {
    throw std::runtime_error("parseNomadJson: results.material is missing.");
  }
  auto material = archive["results"]["material"];

  // Space group symbol if available
  std::string spgSymbol = material.value("symmetry", json())
                                  .value("space_group_symbol", "???");

  // We do not rely on separate a,b,c, alpha,beta,gamma, but might still read volume
  if (!material.contains("topology") || !material["topology"].is_array() ||
      material["topology"].empty()) {
    throw std::runtime_error("parseNomadJson: No topology array found.");
  }
  auto firstTopo = material["topology"][0];
  if (!firstTopo.contains("cell")) {
    throw std::runtime_error("parseNomadJson: 'cell' not in the first topology.");
  }
  auto cellData = firstTopo["cell"];

  double volume = cellData.value("volume", 0.0);
  double density = cellData.value("mass_density", 0.0);

  // Get lattice_vectors from metadata.optimade
  auto optimade = archive["metadata"].value("optimade", json());
  if (!optimade.contains("lattice_vectors") || !optimade["lattice_vectors"].is_array()) {
    throw std::runtime_error("parseNomadJson: 'lattice_vectors' missing or not array.");
  }
  auto lattVecs = optimade["lattice_vectors"];

  // Check 3×3 array
  if (lattVecs.size() != 3) {
    throw std::runtime_error("parseNomadJson: 'lattice_vectors' must have size=3.");
  }
  for (int i = 0; i < 3; i++) {
    if (!lattVecs[i].is_array() || lattVecs[i].size() != 3) {
      throw std::runtime_error("parseNomadJson: Each row in 'lattice_vectors' must have size=3.");
    }
  }

  // Retrieve simplified formula
  std::string formulaReduced = optimade.value("chemical_formula_reduced", "Unknown");

  // Build a descriptive label
  std::ostringstream labelStream;
  labelStream << "NOMAD Entry: " << dataObj.value("entry_id", "???")
              << " | Formula: " << formulaReduced
              << " | SpaceGroup: " << spgSymbol
              << " | Volume: " << volume
              << " | Density: " << density;

  // Assign data to m_structure
  m_structure.label = labelStream.str();

  // Transfer the 3×3 matrix to m_structure.cell
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
}
