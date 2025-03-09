#ifndef TADAH_MLIP_DATA_PROVIDERS_AFLOW_DATA_PROVIDER_H
#define TADAH_MLIP_DATA_PROVIDERS_AFLOW_DATA_PROVIDER_H

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <string>
namespace tadah {
namespace mlip {

/**
 * @brief Provides raw JSON data from AFLOW by an AUID query.
 *
 * Steps:
 *   1) Query "https://aflow.org/API/aflux/?auid('<AUID>')" -> JSON array with "aurl" field.
 *   2) Append "?format=json" to the aurl -> final URL.
 *   3) GET that final URL to obtain the AFLOW structure JSON.
 */
class AFLOWDataProvider : public IMaterialsDataProvider
{
public:
  AFLOWDataProvider() = default;
  ~AFLOWDataProvider() override = default;

  /**
   * @brief Builds a multi-step query using the AUID:
   *        1) "https://aflow.org/API/aflux/?auid('<AUID>')"
   *        2) Final JSON from "aurl"?format=json
   *
   * @param materialID  The AUID string, e.g. "aflow:e9c6d914c4b8d9ca".
   * @return            RawResponse with the final JSON body.
   */
  RawResponse QueryByID(const std::string &materialID) override;

  /**
   * @brief AFLOWDataProvider does not implement formula-based queries in this example.
   */
  RawResponse QueryByFormula(const std::string &formula) override;

private:
  /**
   * @brief cURL write callback, accumulates data in a std::string.
   */
  static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata);

  /**
   * @brief Performs a simple GET request with cURL, returning the complete response body.
   */
  RawResponse httpGet(const std::string &url);

  /**
   * @brief Extracts the "aurl" from the initial AFLOW JSON array, then appends "?format=json".
   *
   * @param auid  The AUID requested.
   * @return      RawResponse with the final JSON from the second request.
   */
  RawResponse fetchAndParseAflowByAuid(const std::string &auid);
};

} // namespace mlip
} // namespace tadah
#endif // TADAH_MLIP_DATA_PROVIDERS_AFLOW_DATA_PROVIDER_H

