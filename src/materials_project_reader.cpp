#include <sstream>
#include <stdexcept>
#include <tadah/mlip/structure_readers/materials_project_reader.h>

#include <curl/curl.h>       // libcurl
#include <nlohmann/json.hpp> // nlohmann JSON

static Element parse_element_mp(const std::string &elemName);

MaterialsProjectReader::MaterialsProjectReader(const std::string &apiKey)
    : m_apiKey(apiKey) {}

void MaterialsProjectReader::read(const std::string &mpID) {
  fetchAndParseMP(mpID);
  m_structure.label = "MaterialsProject ID: " + mpID;
}

Structure MaterialsProjectReader::getStructure() const { return m_structure; }

void MaterialsProjectReader::fetchAndParseMP(const std::string &mpID) {
  std::string url = "https://materialsproject.org/rest/v2/materials/";
  url += mpID + "/structure?API_KEY=" + m_apiKey;

  std::string response = httpGet(url);
  parseMpJson(response);
}

std::string MaterialsProjectReader::httpGet(const std::string &url) {
  CURL *curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error(
        "MaterialsProjectReader::httpGet: Failed to init libcurl");
  }

  auto writeCallback = [](char *ptr, size_t size, size_t nmemb,
                          void *userdata) -> size_t {
    std::string *resp = static_cast<std::string *>(userdata);
    size_t totalSize = size * nmemb;
    resp->append(ptr, totalSize);
    return totalSize;
  };

  std::string response;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    curl_easy_cleanup(curl);
    throw std::runtime_error(
        std::string("httpGet: curl_easy_perform() failed: ") +
        curl_easy_strerror(res));
  }
  curl_easy_cleanup(curl);
  return response;
}

void MaterialsProjectReader::parseMpJson(const std::string &jsonContent) {
  using json = nlohmann::json;
  json root = json::parse(jsonContent);

  if (!root.contains("response") || !root["response"].is_array() ||
      root["response"].empty()) {
    throw std::runtime_error("parseMpJson: No response array in JSON.");
  }

  auto data = root["response"][0];
  if (!data.contains("lattice") || !data["lattice"].contains("matrix")) {
    throw std::runtime_error("parseMpJson: Missing lattice.matrix");
  }

  auto mat = data["lattice"]["matrix"];
  if (!mat.is_array() || mat.size() != 3) {
    throw std::runtime_error("parseMpJson: matrix is not 3x3");
  }

  for (int i = 0; i < 3; i++) {
    if (!mat[i].is_array() || mat[i].size() != 3) {
      throw std::runtime_error("parseMpJson: matrix row is not size 3");
    }
    m_structure.cell(i, 0) = mat[i][0].get<double>();
    m_structure.cell(i, 1) = mat[i][1].get<double>();
    m_structure.cell(i, 2) = mat[i][2].get<double>();
  }

  if (!data.contains("sites") || !data["sites"].is_array()) {
    throw std::runtime_error("parseMpJson: No sites array");
  }
  auto sites = data["sites"];
  for (auto &site : sites) {
    if (!site.contains("abc") || !site["abc"].is_array() ||
        site["abc"].size() != 3) {
      throw std::runtime_error("parseMpJson: site missing abc array");
    }
    double fx = site["abc"][0].get<double>();
    double fy = site["abc"][1].get<double>();
    double fz = site["abc"][2].get<double>();

    // fractional -> cart
    double x = fx * m_structure.cell(0, 0) + fy * m_structure.cell(1, 0) +
               fz * m_structure.cell(2, 0);
    double y = fx * m_structure.cell(0, 1) + fy * m_structure.cell(1, 1) +
               fz * m_structure.cell(2, 1);
    double z = fx * m_structure.cell(0, 2) + fy * m_structure.cell(1, 2) +
               fz * m_structure.cell(2, 2);

    Atom a;
    std::string lbl = site["label"].get<std::string>();
    Element e = parse_element_mp(lbl);
    static_cast<Element &>(a) = e;
    a.position[0] = x;
    a.position[1] = y;
    a.position[2] = z;
    m_structure.atoms.push_back(a);
  }
}

static Element parse_element_mp(const std::string &elemName) {
  return PeriodicTable().find_by_symbol(elemName);
}
