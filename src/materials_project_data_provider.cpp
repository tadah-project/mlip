#include <tadah/mlip/data_providers/materials_project_data_provider.h>
#include <curl/curl.h>
#include <stdexcept>
#include <sstream>
namespace tadah {
namespace mlip {

MaterialsProjectDataProvider::MaterialsProjectDataProvider(const std::string &apiKey)
  : m_apiKey(apiKey)
{
}

RawResponse MaterialsProjectDataProvider::QueryByID(const std::string &materialID)
{
  // Example core materials query
  // Adjust to match current MP endpoints or query parameters
  std::ostringstream url;
  url << "https://api.materialsproject.org/materials/core/"
      << "?material_ids=" << materialID
      << "&deprecated=false"
      << "&_all_fields=true"
      << "&license=BY-NC";

  return httpGet(url.str());
}

RawResponse MaterialsProjectDataProvider::QueryByFormula(const std::string &formula)
{
  // Hypothetical formula-based query. Adjust as needed for actual MP endpoints.
  std::ostringstream url;
  url << "https://api.materialsproject.org/materials/core/"
      << "?formula=" << formula
      << "&deprecated=false"
      << "&_all_fields=true"
      << "&license=BY-NC";

  return httpGet(url.str());
}

size_t MaterialsProjectDataProvider::writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* response = static_cast<std::string*>(userdata);
  size_t totalBytes = size * nmemb;
  response->append(ptr, totalBytes);
  return totalBytes;
}

RawResponse MaterialsProjectDataProvider::httpGet(const std::string &url)
{
  RawResponse rr;
  rr.httpStatus = 400;
  rr.response.clear();

  CURL* curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("MaterialsProjectDataProvider: cURL init failed.");
  }

  struct curl_slist* headers = nullptr;
  {
    std::ostringstream hdr;
    hdr << "X-API-KEY: " << m_apiKey;
    headers = curl_slist_append(headers, hdr.str().c_str());
  }
  headers = curl_slist_append(headers, "Accept: application/json");

  std::string buffer;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  // Optional user-agent
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "Tadah/MLIP: MaterialsProjectDataProvider");

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  rr.httpStatus = static_cast<int>(http_code);

  if (res != CURLE_OK) {
    rr.httpStatus = 500;
    std::ostringstream oss;
    oss << "MaterialsProjectDataProvider: cURL error: " << curl_easy_strerror(res);
    rr.response = oss.str();
  } else {
    rr.response = buffer;
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return rr;
}

} // namespace mlip
} // namespace tadah
