#ifndef TADAH_MLIP_DATA_PROVIDERS_IMATERIALS_DATA_PROVIDER_H
#define TADAH_MLIP_DATA_PROVIDERS_IMATERIALS_DATA_PROVIDER_H

#include <string>
namespace tadah {
namespace mlip {

/**
 * @brief Encapsulates the raw result of an HTTP request to a materials-database service.
 *        Provides the status code and the unparsed body (JSON, text, etc.).
 */
struct RawResponse
{
  /**
   * @brief HTTP status code (e.g., 200 for success, 404 for not found).
   */
  int httpStatus;

  /**
   * @brief The server's response data as plain text (often JSON or XML).
   */
  std::string response;
};

/**
 * @brief An interface for querying various materials-data services
 *        such as COD, NOMAD, AFLOW, or Materials Project.
 *
 * Implementations handle only basic HTTP or REST interactions,
 * returning RawResponse objects. No parsing of JSON or other formats
 * is performed at this level.
 */
class IMaterialsDataProvider
{
public:
  /// Virtual destructor ensures correct cleanup of derived classes.
  virtual ~IMaterialsDataProvider() = default;

  /**
   * @brief Retrieves unparsed data for a single material by a service-specific ID.
   *
   * For COD, this might be a 7-digit numeric ID (e.g., "1534932"),
   * for AFLOW, an AUID (e.g., "aflow:e9c6d914c4b8d9ca"),
   * or for the Materials Project, "mp-149".
   *
   * @param materialID  The service-specific material identifier.
   * @return            A RawResponse with the HTTP code and the raw body.
   */
  virtual RawResponse QueryByID(const std::string &materialID) = 0;

  /**
   * @brief Retrieves unparsed data by formula (e.g., "Fe2O3" or "SiO2"),
   *        if the service supports formula-based queries.
   *
   * @param formula  A chemical formula string.
   * @return         A RawResponse with the HTTP code and the raw body.
   */
  virtual RawResponse QueryByFormula(const std::string &formula) = 0;

  /**
   * @brief Potential future expansions (6.1, 6.2):
   *        e.g., advanced filtering, batch queries, or custom property requests.
   *
   * Methods may be added here in the future, for example:
   *
   *   // virtual RawResponse QueryByElementSet(const std::vector<std::string>& elements) = 0;
   *   // virtual RawResponse QueryBatch(const std::vector<std::string>& ids) = 0;
   */

  // Additional placeholders can be added for asynchronous operations (6.5).
};

} // namespace mlip
} // namespace tadah
#endif // TADAH_MLIP_DATA_PROVIDERS_IMATERIALS_DATA_PROVIDER_H

