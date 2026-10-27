#include <tadah/mlip/data_providers/cod_data_provider.h>
#include <curl/curl.h>
#include <stdexcept>
#include <sstream>
namespace tadah {
namespace mlip {

size_t CODDataProvider::writeDataCallback(void* ptr, size_t size, size_t nmemb, void* userData)
{
  size_t totalSize = size * nmemb;
  auto* str = static_cast<std::string*>(userData);
  str->append(static_cast<char*>(ptr), totalSize);
  return totalSize;
}

RawResponse CODDataProvider::QueryByID(const std::string &materialID)
{
  // Build final .cif URL
  std::string url = "https://www.crystallography.net/cod/";
  if (materialID.find(".cif") == std::string::npos) {
    url += materialID + ".cif";
  } else {
    url += materialID;
  }

  return fetchFromURL(url);
}

RawResponse CODDataProvider::QueryByFormula(const std::string &formula)
{
  // COD has no trivial single-endpoint for formula queries.
  RawResponse result;
  result.httpStatus = 501; // 'Not Implemented'
  result.response = "COD: formula queries are not supported.";
  return result;
}

RawResponse CODDataProvider::fetchFromURL(const std::string &url)
{
  RawResponse rr;
  rr.httpStatus = 400; // Default to bad request if something fails
  rr.response.clear();

  CURL* curlHandle = curl_easy_init();
  if (!curlHandle) {
    throw std::runtime_error("COD: cURL does not initialise.");
  }

  // Prepare string buffer for the response
  std::string buffer;
  curl_easy_setopt(curlHandle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, writeDataCallback);
  curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &buffer);

  // Perform request
  CURLcode curlRes = curl_easy_perform(curlHandle);

  // Retrieve HTTP status code
  long http_code = 0;
  curl_easy_getinfo(curlHandle, CURLINFO_RESPONSE_CODE, &http_code);
  rr.httpStatus = static_cast<int>(http_code);

  if (curlRes != CURLE_OK) {
    rr.httpStatus = 500; // Internal error
    std::ostringstream oss;
    oss << "COD: cURL error: " << curl_easy_strerror(curlRes);
    rr.response = oss.str();
  } else {
    rr.response = buffer;
  }

  curl_easy_cleanup(curlHandle);
  return rr;
}

} // namespace mlip
} // namespace tadah
