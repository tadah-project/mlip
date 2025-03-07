#ifndef TADAH_MLIP_DATA_PROVIDERS_COD_DATA_PROVIDER_H
#define TADAH_MLIP_DATA_PROVIDERS_COD_DATA_PROVIDER_H

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <string>

/**
 * @brief Provides raw CIF data from the Crystallography Open Database (COD)
 *        without performing any parsing.
 *
 * QueryByID downloads a .cif file based on a numerical COD ID (e.g. "1534932").
 * If 'codID' lacks a ".cif" suffix, ".cif" is appended automatically.
 */
class CODDataProvider : public IMaterialsDataProvider
{
public:
  /// Default constructor. No authentication is required for COD.
  CODDataProvider() = default;

  /**
   * @brief Implementation of QueryByID for COD.
   *        Constructs a URL like:
   *          "https://www.crystallography.net/cod/1534932.cif"
   *
   * @param materialID  The COD ID (e.g. "1534932").
   * @return            A RawResponse with the HTTP status code and CIF text.
   */
  RawResponse QueryByID(const std::string &materialID) override;

  /**
   * @brief Implementation of QueryByFormula for COD.
   *        COD does not provide a simple formula-based endpoint, so returns 501.
   *
   * @param formula  The chemical formula (e.g. "NaCl").
   * @return         A RawResponse with httpStatus=501 and a message.
   */
  RawResponse QueryByFormula(const std::string &formula) override;

private:
  /**
   * @brief Receives data via libcurl's write callback into a std::string.
   *
   * @param ptr        Pointer to received bytes.
   * @param size       Size of each element.
   * @param nmemb      Number of elements.
   * @param userData   Pointer to the string to be appended.
   * @return           Number of bytes actually taken.
   */
  static size_t writeDataCallback(void* ptr, size_t size, size_t nmemb, void* userData);

  /**
   * @brief Performs a GET request to the specified URL using libcurl.
   *        Records HTTP status and the entire response body.
   *
   * @param url   The COD target URL (e.g. "https://www.crystallography.net/cod/1534932.cif").
   * @return      A RawResponse with an HTTP status code and the body text.
   */
  RawResponse fetchFromURL(const std::string &url);
};

#endif // TADAH_MLIP_DATA_PROVIDERS_COD_DATA_PROVIDER_H

