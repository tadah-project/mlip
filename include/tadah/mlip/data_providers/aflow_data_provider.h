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
 *   2) Build the entry file's URL from the aurl (entry_url_from_aflux):
 *      "server:PATH" -> "http://server/PATH/aflowlib.json".
 *   3) GET that URL to obtain the AFLOW entry JSON (typed fields; see
 *      AflowReader::parseAflowJson).
 */
class AFLOWDataProvider : public IMaterialsDataProvider
{
public:
  AFLOWDataProvider() = default;
  ~AFLOWDataProvider() override = default;

  /**
   * @brief Builds a multi-step query using the AUID:
   *        1) "https://aflow.org/API/aflux/?auid('<AUID>')"
   *        2) Final JSON from the entry file <aurl>/aflowlib.json
   *
   * @param materialID  The AUID string, e.g. "aflow:e9c6d914c4b8d9ca".
   * @return            RawResponse with the final JSON body; HTTP 404, with
   *                    the reason as the response, when the AFLUX answer
   *                    gives no entry URL.
   */
  RawResponse QueryByID(const std::string &materialID) override;

  /**
   * @brief AFLOWDataProvider does not implement formula-based queries in this example.
   */
  RawResponse QueryByFormula(const std::string &formula) override;

  /**
   * @brief The URL of the entry file from an AFLUX answer: the "aurl" of
   *        its first entry, "server:PATH", as
   *        "http://server/PATH/aflowlib.json".
   *
   * For "aflowlib.duke.edu:AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152" it is
   * "http://aflowlib.duke.edu/AFLOWDATA/ICSD_WEB/BCC/Ta1_ICSD_76152/aflowlib.json".
   * The entry's directory URL with "?format=json" is answered with 404.
   *
   * @param aflux_json  The body of the AFLUX query (step 1).
   * @throws std::runtime_error when the answer is not a non-empty JSON
   *         array whose first element is an object with a string "aurl",
   *         or when that aurl is not "server:PATH".
   */
  static std::string entry_url_from_aflux(const std::string &aflux_json);

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
   * @brief Queries AFLUX for the AUID, then GETs the entry file whose URL
   *        entry_url_from_aflux builds from the answer.
   *
   * @param auid  The AUID requested.
   * @return      RawResponse with the final JSON from the second request,
   *              the first request's response when its status is not 200,
   *              or HTTP 404 with the reason when its answer gives no URL.
   */
  RawResponse fetchAndParseAflowByAuid(const std::string &auid);
};

} // namespace mlip
} // namespace tadah
#endif // TADAH_MLIP_DATA_PROVIDERS_AFLOW_DATA_PROVIDER_H

