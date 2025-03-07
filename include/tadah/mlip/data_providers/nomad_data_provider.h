#ifndef TADAH_MLIP_DATA_PROVIDERS_NOMAD_DATA_PROVIDER_H
#define TADAH_MLIP_DATA_PROVIDERS_NOMAD_DATA_PROVIDER_H

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <string>

/**
 * @brief Provides raw data from NOMAD by making HTTP POST requests to:
 *        "https://nomad-lab.eu/prod/v1/api/v1/entries/<entryID>/archive/query".
 *
 * No JSON parsing is performed; the raw response is returned to the caller.
 */
class NomadDataProvider : public IMaterialsDataProvider
{
public:
  /// Default constructor (no special authentication required).
  NomadDataProvider() = default;
  ~NomadDataProvider() override = default;

  /**
   * @brief Uses the NOMAD entry ID to form a POST request to fetch raw metadata.
   *
   * Example endpoint:
   *   "https://nomad-lab.eu/prod/v1/api/v1/entries/<entryID>/archive/query"
   *
   * @param materialID The NOMAD entry identifier (e.g. "zyXabc123...").
   * @return           HTTP code and raw JSON in RawResponse.
   */
  RawResponse QueryByID(const std::string &materialID) override;

  /**
   * @brief NOMAD does not provide a straightforward single-endpoint formula-based search
   *        at the moment, so returns 501 Not Implemented.
   */
  RawResponse QueryByFormula(const std::string &formula) override;

private:
  /**
   * @brief Collects POST response data into a std::string.
   */
  static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata);

  /**
   * @brief Performs the NOMAD POST request with a fixed JSON body requesting
   *        topology and structure metadata for the given entry ID.
   *
   * @param materialID The NOMAD entry ID.
   * @return           RawResponse with HTTP status and the body text.
   */
  RawResponse fetchNomadArchive(const std::string &materialID);

  /**
   * @brief Sends a POST request with JSON to the specified URL,
   *        returning the unparsed response.
   *
   * @param url       The endpoint for the POST request.
   * @param jsonBody  The JSON payload to send.
   * @return          RawResponse with status code and body.
   */
  RawResponse httpPost(const std::string &url, const std::string &jsonBody);
};

#endif // TADAH_MLIP_DATA_PROVIDERS_NOMAD_DATA_PROVIDER_H

