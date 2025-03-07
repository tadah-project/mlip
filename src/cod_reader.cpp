#include <tadah/mlip/structure_readers/cod_reader.h>

#include <curl/curl.h>
#include <stdexcept>
#include <string>

/*
  writeDataCallback() is used by libcurl to write received bytes
  into a std::string pointed to by userData.
*/
size_t codReader::writeDataCallback(void* ptr, size_t size, size_t nmemb, void* userData) {
  size_t totalSize = size * nmemb;
  auto* str = static_cast<std::string*>(userData);
  str->append(static_cast<char*>(ptr), totalSize);
  return totalSize;
}

/*
  read() determines whether the given path is a URL or a COD entry ID,
  constructs the proper URL if needed, then downloads the CIF data as a string.
  The data is parsed using an existing CifReader, and stored in m_structure.
*/
void codReader::read(const std::string &path) {
  // Determines final URL to query
  std::string codUrl;
  if (path.find("http://") == std::string::npos &&
      path.find("https://") == std::string::npos) {
    // If no ".cif" suffix found, append it
    if (path.find(".cif") == std::string::npos) {
      codUrl = "https://www.crystallography.net/cod/" + path + ".cif";
    } else {
      codUrl = "https://www.crystallography.net/cod/" + path;
    }
  } else {
    // Already a valid URL
    codUrl = path;
  }

  // Uses libcurl to download the CIF data as a string
  const std::string cifData = fetchDataFromURL(codUrl);

  // Uses existing CifReader to parse the CIF data
  CifReader creader;
  creader.parseCifContents(cifData);

  // Retrieves the parsed Structure
  m_structure = creader.getStructure();
  m_structure.label.insert(0,"COD entry: " + path);
}

/*
  getStructure() returns the most recently read structure for usage 
  elsewhere in the codebase.
*/
Structure codReader::getStructure() const {
  return m_structure;
}

/*
  fetchDataFromURL() configures libcurl to make an HTTP GET request
  and returns the response content.
*/
std::string codReader::fetchDataFromURL(const std::string &url) const {
  CURL* curlHandle = curl_easy_init();
  if (!curlHandle) {
    throw std::runtime_error("Failed to initialize libcurl.");
  }

  // Sets the URL and follows any redirection
  curl_easy_setopt(curlHandle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, 1L);

  // Prepares string buffer for collected data
  std::string buffer;
  curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, writeDataCallback);
  curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &buffer);

  // Executes the request
  CURLcode result = curl_easy_perform(curlHandle);
  curl_easy_cleanup(curlHandle);

  // Checks for errors during transfer
  if (result != CURLE_OK) {
    throw std::runtime_error(std::string("Data transfer failed: ") +
                             curl_easy_strerror(result));
  }
  return buffer;
}

