#ifndef TADAH_MLIP_DATA_PROVIDERS_MATERIALS_PROJECT_DATA_PROVIDER_H
#define TADAH_MLIP_DATA_PROVIDERS_MATERIALS_PROJECT_DATA_PROVIDER_H

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <string>
namespace tadah {
namespace mlip {

/**
 * @brief Retrieves raw JSON from the Materials Project API (v2 / v3).
 *
 * Authentication is handled via an API key passed in the X-API-KEY header.
 */
class MaterialsProjectDataProvider : public IMaterialsDataProvider
{
public:
  /**
   * @brief Constructor storing the user-provided Materials Project API key.
   *
   * @param apiKey A valid Materials Project key, e.g. "1234567890ABC".
   */
  explicit MaterialsProjectDataProvider(const std::string &apiKey);

  /**
   * @brief Query by ID, e.g. "mp-149".
   *        Construct the appropriate URL and attach the X-API-KEY header.
   */
  RawResponse QueryByID(const std::string &materialID) override;

  /**
   * @brief Query by formula (e.g., "Fe2O3") if the endpoint supports formula-based lookups.
   */
  RawResponse QueryByFormula(const std::string &formula) override;

private:
  std::string m_apiKey;

  /**
   * @brief Callback for storing downloaded data into a std::string.
   */
  static size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata);

  /**
   * @brief Executes a GET request to the provided URL using libcurl,
   *        adding the X-API-KEY header with m_apiKey.
   */
  RawResponse httpGet(const std::string &url);
};

} // namespace mlip
} // namespace tadah
#endif // TADAH_MLIP_DATA_PROVIDERS_MATERIALS_PROJECT_DATA_PROVIDER_H

