/**
  MaterialsProjectReader.cpp
  Implements functionality declared in MaterialsProjectReader.h.

  Example usage:
  --------------------------------------------------------------------
  // 1) The request sets multiple query parameters:
  //      material_ids=mp-35
  //      deprecated=false
  //      _per_page=100
  //      _skip=0
  //      _limit=100
  //      _all_fields=true
  //      license=BY-NC
  //
  // 2) The request sets the header:
  //      X-API-KEY: <some-api-key>
  //
  // 3) The server is expected to return JSON data with an array
  //    under "data", each item a "MaterialsDoc" from which we parse
  //    the "structure" field's lattice and sites.
  //
  // 4) This code is integrated with the rest of the codebase by
  //    using the MaterialsProjectReader constructor to store the
  //    API key. The read() method then forms the correct URL
  //    with the additional query parameters and a custom header.
*/

#include <tadah/mlip/structure_readers/materials_project_reader.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <iostream>

// parse_element_mp helps parse site labels or species info to retrieve
// an Element from a known PeriodicTable utility.
static Element parse_element_mp(const std::string &elemName) {
  return PeriodicTable().find_by_symbol(elemName);
}

// Constructor: stores user-provided Materials Project API key.
MaterialsProjectReader::MaterialsProjectReader(const std::string &apiKey)
  : m_apiKey(apiKey) {
}

// read fetches and parses structure data for a given mpID (e.g. "mp-35").
void MaterialsProjectReader::read(const std::string &mpID) {
  fetchAndParseMP(mpID);
}

// getStructure returns the last structure retrieved.
Structure MaterialsProjectReader::getStructure() const {
  return m_structure;
}

// fetchAndParseMP forms a query URL that includes the user example query params,
// calls httpGet with a relevant header, then parses the JSON in parseMpJson.
void MaterialsProjectReader::fetchAndParseMP(const std::string &mpID) {
  const std::string url =
    "https://api.materialsproject.org/materials/core/"
    "?material_ids="
    + mpID +
    "&deprecated=false"
    "&_per_page=100"
    "&_skip=0"
    "&_limit=100"
    "&_all_fields=true"
    "&license=BY-NC";

  // Execute an HTTP GET request via cURL
  std::string response = httpGet(url);

  // Parse the JSON response
  parseMpJson(response);
}
// This callback accumulates data in a std::string.
size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* response = static_cast<std::string*>(userdata);
  size_t totalBytes = size * nmemb;
  response->append(ptr, totalBytes);
  return totalBytes;
}

// httpGet uses cURL to perform an HTTP GET on `url`, including
// the X-API-KEY header matching the Materials Project specification.
std::string MaterialsProjectReader::httpGet(const std::string &url) {

  // cURL initialization
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "Error: Failed to init cURL\n";
  }

  // The server response will be stored here.
  std::string response;

  // Prepare custom headers:
  struct curl_slist* headers = nullptr;
  {
    std::stringstream ss;
    ss << "X-API-KEY: " << m_apiKey;
    headers = curl_slist_append(headers, ss.str().c_str());
  }
  headers = curl_slist_append(headers, "accept: application/json");

  // cURL basic configuration:
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  // User-Agent string is required by the Materials Project API
  // to identify the client. Otherwise, the request will be rejected.
  curl_easy_setopt(curl, CURLOPT_USERAGENT,
    "Tadah! (https://tadah.readthedocs.io)");

  // The write callback to collect the HTTP response data.
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  // Follow redirects if needed
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  // Perform the HTTP GET
  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::cerr << "[DEBUG] cURL perform failed with error code: " << res << std::endl;
    // Cleanup on error
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    throw std::runtime_error(
      std::string("httpGet: curl_easy_perform() failed: ") +
      curl_easy_strerror(res));
  }

  // Cleanup
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return response;
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

  if (!root.contains("data") || !root["data"].is_array() ||
      root["data"].empty()) {
    throw std::runtime_error("parseMpJson: Missing or empty 'data' array.");
  }

  auto doc = root["data"][0];
  if (!doc.contains("structure")) {
    throw std::runtime_error("parseMpJson: 'structure' not found.");
  }

  auto structureJson = doc["structure"];
  if (!structureJson.contains("lattice") ||
      !structureJson["lattice"].contains("matrix")) {
    throw std::runtime_error("parseMpJson: Missing 'lattice.matrix'.");
  }

  // Build a label for the structure
  auto symbol = to_string(doc["symmetry"]["symbol"]);
  auto volume = to_string(doc["volume"]);
  auto density = to_string(doc["density"]);
  auto elements = to_string(doc["elements"]);
  auto mid = to_string(doc["material_id"]);
  m_structure.label = "MaterialsProject ID: " + mid+ " | " +
                      "Symmetry: " + symbol+ " | " +
                      "Volume: " + volume+ " | " +
                      "Density: " + density+ " | " +
                      "Elements: " + elements+ " | ";

  // Extract the 3x3 lattice matrix
  auto mat = structureJson["lattice"]["matrix"];
  if (!mat.is_array() || mat.size() != 3) {
    throw std::runtime_error("parseMpJson: 'matrix' must be array of length 3.");
  }
  for (int i = 0; i < 3; i++) {
    if (!mat[i].is_array() || mat[i].size() != 3) {
      throw std::runtime_error("parseMpJson: matrix row must be length 3.");
    }
    m_structure.cell(i, 0) = mat[i][0].get<double>();
    m_structure.cell(i, 1) = mat[i][1].get<double>();
    m_structure.cell(i, 2) = mat[i][2].get<double>();
  }

  // 'sites' array: each site has fractional coordinates "abc"
  if (!structureJson.contains("sites") || !structureJson["sites"].is_array()) {
    throw std::runtime_error("parseMpJson: 'sites' is missing or not an array.");
  }
  auto sites = structureJson["sites"];

  for (auto &site : sites) {
    if (!site.contains("abc") || !site["abc"].is_array() ||
        site["abc"].size() != 3) {
      throw std::runtime_error("parseMpJson: Site missing valid 'abc'.");
    }
    double fx = site["abc"][0].get<double>();
    double fy = site["abc"][1].get<double>();
    double fz = site["abc"][2].get<double>();

    // Convert fractional -> cart
    double x = fx * m_structure.cell(0, 0) +
               fy * m_structure.cell(1, 0) +
               fz * m_structure.cell(2, 0);
    double y = fx * m_structure.cell(0, 1) +
               fy * m_structure.cell(1, 1) +
               fz * m_structure.cell(2, 1);
    double z = fx * m_structure.cell(0, 2) +
               fy * m_structure.cell(1, 2) +
               fz * m_structure.cell(2, 2);

    // The element can be found in either 'label' or 'species[0].element'
    std::string elemSymbol;
    if (site.contains("label") && site["label"].is_string()) {
      elemSymbol = site["label"].get<std::string>();
    } else if (site.contains("species") && site["species"].is_array() &&
               !site["species"].empty() &&
               site["species"][0].contains("element")) {
      elemSymbol = site["species"][0]["element"].get<std::string>();
    } else {
      throw std::runtime_error(
        "parseMpJson: No recognized element data in site.");
    }

    Atom a;
    Element e = parse_element_mp(elemSymbol);
    static_cast<Element &>(a) = e;
    a.position[0] = x;
    a.position[1] = y;
    a.position[2] = z;

    m_structure.atoms.push_back(a);
  }
}
