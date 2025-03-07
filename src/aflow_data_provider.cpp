#include <tadah/mlip/data_providers/aflow_data_provider.h>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <stdexcept>
#include <sstream>

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
  r.response = "AFLOWDataProvider does not implement formula queries.";
  return r;
}

RawResponse AFLOWDataProvider::httpGet(const std::string &url)
{
  RawResponse rr;
  rr.httpStatus = 400;
  rr.response.clear();

  CURL* curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("AFLOWDataProvider: cURL init failed.");
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
    oss << "AFLOWDataProvider: cURL error: " << curl_easy_strerror(res);
    rr.response = oss.str();
  } else {
    rr.response = response;
  }

  curl_easy_cleanup(curl);
  return rr;
}

/*
  convertAURLToURL(aurl):
    Converts an AFLOW-style aurl, usually "server:PATH", 
    into "server/PATH". This helps build a domain/path 
    format recognized by HTTP requests. For example:

      "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152"

    becomes:

      "aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152".
*/
std::string convertAURLToURL(const std::string &aurl) {
  const std::size_t pos = aurl.find(':');
  if (pos != std::string::npos) {
    std::string server = aurl.substr(0, pos);
    std::string path   = aurl.substr(pos + 1);
    return server + "/" + path;
  }
  return aurl;
}

RawResponse AFLOWDataProvider::fetchAndParseAflowByAuid(const std::string &auid)
{
  // Step 1) Query the AFLOW "aflux" endpoint for the "aurl"
  // "https://aflow.org/API/aflux/?auid('<auid>')"
  RawResponse step1 = httpGet("https://aflow.org/API/aflux/?auid('" + auid + "')");
  if (step1.httpStatus != 200) {
    return step1; // Return directly if the first step fails
  }

  // Minimal parse to extract "aurl" from the result
  // This is the only partial parse required to get the second URL.
  // The final data is still returned unparsed from step2.
  nlohmann::json root = nlohmann::json::parse(step1.response, nullptr, false);
  if (!root.is_array() || root.empty()) {
    RawResponse rr;
    rr.httpStatus = 404;
    rr.response = "AFLOWDataProvider: 'aflux' JSON not an array or is empty.";
    return rr;
  }

  // We assume the first object's "aurl" field is needed
  auto firstObj = root[0];
  if (!firstObj.contains("aurl")) {
    RawResponse rr;
    rr.httpStatus = 404;
    rr.response = "AFLOWDataProvider: 'aurl' field not found in JSON.";
    return rr;
  }

  std::string aurl = firstObj["aurl"].get<std::string>();
  aurl = convertAURLToURL(aurl);
  if (aurl.find('?') == std::string::npos) {
    // Append "?format=json" if not present
    aurl += "?format=json";
  } else {
    aurl += "&format=json";
  }

  // Step 2) GET final data from 'aurl'
  return httpGet(aurl);
}
