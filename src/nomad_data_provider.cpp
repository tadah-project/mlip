#include <tadah/mlip/data_providers/nomad_data_provider.h>
#include <curl/curl.h>
#include <stdexcept>
#include <sstream>
namespace tadah {
namespace mlip {

size_t NomadDataProvider::writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* response = static_cast<std::string*>(userdata);
  size_t totalBytes = size * nmemb;
  response->append(ptr, totalBytes);
  return totalBytes;
}

RawResponse NomadDataProvider::QueryByID(const std::string &materialID)
{
  return fetchNomadArchive(materialID);
}

RawResponse NomadDataProvider::QueryByFormula(const std::string &formula)
{
  RawResponse r;
  r.httpStatus = 501; 
  r.response = "NOMAD: formula queries are not supported.";
  return r;
}

RawResponse NomadDataProvider::fetchNomadArchive(const std::string &materialID)
{
  // Construct the URL
  const std::string baseURL = "https://nomad-lab.eu/prod/v1/api/v1";
  const std::string url = baseURL + "/entries/" + materialID + "/archive/query";

  // Minimal example JSON body (the full query can be adapted if needed).
  static const std::string requestBody = R"JSON(
  {
    "required": {
      "results": { "material": { "symmetry": "*", "topology": "*" } },
      "metadata": { "optimade": "*" }
    }
  }
  )JSON";

  return httpPost(url, requestBody);
}

RawResponse NomadDataProvider::httpPost(const std::string &url, const std::string &jsonBody)
{
  RawResponse rr;
  rr.httpStatus = 400;
  rr.response.clear();

  CURL* curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("NOMAD: cURL does not initialise.");
  }

  std::string response;
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Accept: application/json");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonBody.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(jsonBody.size()));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  CURLcode res = curl_easy_perform(curl);

  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  rr.httpStatus = static_cast<int>(http_code);

  if (res != CURLE_OK) {
    rr.httpStatus = 500;
    std::ostringstream oss;
    oss << "NOMAD: cURL error: " << curl_easy_strerror(res);
    rr.response = oss.str();
  } else {
    rr.response = response;
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  return rr;
}

} // namespace mlip
} // namespace tadah
