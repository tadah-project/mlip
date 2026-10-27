#include <tadah/mlip/data_providers/aflow_data_provider.h>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <stdexcept>
#include <sstream>
namespace tadah {
namespace mlip {

size_t AFLOWDataProvider::writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* response = static_cast<std::string*>(userdata);
  size_t totalBytes = size * nmemb;
  response->append(ptr, totalBytes);
  return totalBytes;
}

RawResponse AFLOWDataProvider::QueryByID(const std::string &materialID)
{
  return fetchAndParseAflowByAuid(materialID);
}

RawResponse AFLOWDataProvider::QueryByFormula(const std::string &formula)
{
  RawResponse r;
  r.httpStatus = 501;
  r.response = "AFLOW: formula queries are not supported.";
  return r;
}

RawResponse AFLOWDataProvider::httpGet(const std::string &url)
{
  RawResponse rr;
  rr.httpStatus = 400;
  rr.response.clear();

  CURL* curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("AFLOW: cURL does not initialise.");
  }

  std::string response;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "AFLOWDataProvider/1.0");
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  rr.httpStatus = static_cast<int>(http_code);

  if (res != CURLE_OK) {
    rr.httpStatus = 500;
    std::ostringstream oss;
    oss << "AFLOW: cURL error: " << curl_easy_strerror(res);
    rr.response = oss.str();
  } else {
    rr.response = response;
  }

  curl_easy_cleanup(curl);
  return rr;
}

/*
  entry_url_from_aflux(aflux_json):
    The AFLUX answer is a JSON array of entries; the "aurl" of the first,
    "server:PATH", locates the entry's directory, and the entry file in it
    is aflowlib.json. For example

      "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152"

    becomes

      "http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152/aflowlib.json".

    The directory URL with "?format=json", requested before, is answered
    with 404 "format=json not available for this entry" (checked on
    2026-10-06).
*/
std::string AFLOWDataProvider::entry_url_from_aflux(const std::string &aflux_json)
{
  const nlohmann::json root = nlohmann::json::parse(aflux_json, nullptr, false);
  if (!root.is_array() || root.empty()) {
    throw std::runtime_error(
        "AFLOW: the AFLUX answer is not a non-empty JSON array.");
  }
  const nlohmann::json &first = root[0];
  if (!first.is_object() || !first.contains("aurl") ||
      !first["aurl"].is_string()) {
    throw std::runtime_error(
        "AFLOW: the first entry of the AFLUX answer has no "
        "\"aurl\" string.");
  }
  const std::string aurl = first["aurl"].get<std::string>();
  const std::size_t colon = aurl.find(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 == aurl.size()) {
    throw std::runtime_error("AFLOW: aurl '" + aurl +
                             "' in the AFLUX answer is not server:PATH.");
  }
  return "http://" + aurl.substr(0, colon) + "/" + aurl.substr(colon + 1) +
         "/aflowlib.json";
}

RawResponse AFLOWDataProvider::fetchAndParseAflowByAuid(const std::string &auid)
{
  // Step 1) Query the AFLOW "aflux" endpoint for the "aurl"
  // "https://aflow.org/API/aflux/?auid('<auid>')"
  RawResponse step1 = httpGet("https://aflow.org/API/aflux/?auid('" + auid + "')");
  if (step1.httpStatus != 200) {
    return step1; // Return directly if the first step fails
  }

  // The only partial parse required: the entry file's URL. An answer
  // without one gives 404 with the reason in the response; the entry
  // itself is still returned unparsed from step 2.
  std::string url;
  try {
    url = entry_url_from_aflux(step1.response);
  } catch (const std::runtime_error &e) {
    RawResponse rr;
    rr.httpStatus = 404;
    rr.response = e.what();
    return rr;
  }

  // Step 2) GET the entry file <aurl>/aflowlib.json
  return httpGet(url);
}
} // namespace mlip
} // namespace tadah
