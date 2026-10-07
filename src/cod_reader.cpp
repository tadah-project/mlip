#include <tadah/mlip/structure_readers/cod_reader.h>
#include <tadah/mlip/data_providers/fetch_failure.h>
#include <stdexcept>
#include <algorithm>
namespace tadah {
namespace mlip {

CODReader::CODReader()
  : m_dataProvider(std::make_unique<CODDataProvider>()) {
}

CODReader::CODReader(std::unique_ptr<IMaterialsDataProvider> provider)
  : m_dataProvider(std::move(provider)) {
  if (!m_dataProvider)
    throw std::invalid_argument("CODReader: the data provider is null.");
}

bool CODReader::isHttpUrl(const std::string &path) const {
  if (path.size() < 7) { 
    return false; 
  }
  // Minimal check: starts with http:// or https://
  const std::string lower = 
    (path.size() >= 8) ? std::string(path.begin(), path.begin() + 8) : path;
  return (lower.rfind("http://", 0) == 0 ||
          lower.rfind("https://", 0) == 0);
}

std::string CODReader::extractCodIdFromUrl(const std::string &url) const {
  // Example URL: "https://www.crystallography.net/cod/1534932.cif"
  // We want to extract "1534932" if possible:
  //  1) find "/cod/" => everything after that, up to ".cif"
  auto pos = url.find("/cod/");
  if (pos == std::string::npos) {
    return std::string(); // not found
  }
  pos += 5; // skip "/cod/"
  auto dotPos = url.find(".cif", pos);
  if (dotPos == std::string::npos) {
    return std::string();
  }
  return url.substr(pos, dotPos - pos);
}

std::string CODReader::fetchCifById(const std::string &codID) const {
  RawResponse rr = m_dataProvider->QueryByID(codID);
  if (rr.httpStatus != 200)
    throw std::runtime_error(
        failed_fetch_message("CODReader", "COD ID", codID, rr));
  return rr.response; // The raw CIF text
}

void CODReader::read(const std::string &path) {
  // Decide whether path is a URL or an ID:
  std::string codID;
  if (isHttpUrl(path)) {
    codID = extractCodIdFromUrl(path);
    if (codID.empty()) {
      throw std::runtime_error("CODReader::read - Cannot parse COD ID from URL: " + path);
    }
  } else {
    if(path.rfind("cod-", 0) == 0) {
      // If it starts with "cod-", strip that prefix
      codID = path.substr(4); // skip "cod-"
    } else {
      codID = path; // assume direct numeric ID
    } 
  }

  // Fetch raw CIF from COD
  std::string cifData = fetchCifById(codID);

  // Parse CIF data with CifReader
  CifReader creader;
  creader.parseCifContents(cifData);
  m_structure = creader.getStructure();

  // Annotate structure label
  m_structure.label.insert(0, "COD entry: " + path + " | ");
}

Structure CODReader::getStructure() const {
  return m_structure;
}
} // namespace mlip
} // namespace tadah
