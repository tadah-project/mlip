#ifndef COD_READER_H
#define COD_READER_H

#include <memory>
#include <string>

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <tadah/mlip/data_providers/cod_data_provider.h>
#include <tadah/mlip/structure_readers/istructure_reader.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure.h>
namespace tadah {
namespace mlip {

/**
 * @brief Reads a CIF file from the Crystallography Open Database (COD)
 *        and parses its content using CifReader.
 *
 * Delegates all network fetching to CODDataProvider, then uses CifReader
 * to convert raw CIF text into a Structure.
 */
class CODReader : public IStructureReader {
public:
  /// Default constructor
  CODReader();

  /**
   * @brief A CODReader that fetches through @p provider (a stub in tests);
   *        a null provider is refused (std::invalid_argument).
   */
  explicit CODReader(std::unique_ptr<IMaterialsDataProvider> provider);

  /**
   * @brief Retrieves a CIF resource from the COD database by ID or by a full URL,
   *        then parses it into a Structure.
   *
   * If @p path is recognized as a URL (e.g. starts with "http://" or "https://"),
   * the code attempts to extract the COD material ID. Otherwise, it uses @p path
   * directly as the COD ID. The returned raw content is parsed with CifReader.
   *
   * A failed fetch throws std::runtime_error naming the HTTP status (or the
   * provider's own: 500 for a cURL error such as no network), the COD ID
   * (without "cod-") and the provider's reason, cut to 300 characters
   * (failed_fetch_message(): `CODReader: HTTP 404 for COD ID '1534932':
   * <the server's page>`). The reason is the provider's cURL text or the
   * server's body; the message holds no request URL. The previous structure
   * is kept.
   *
   * @param path  A COD entry ID (optionally with prepended "cod-")
   * (e.g. "cod-1534932" and "1534932" are both valid)
   * or a fully qualified URL (e.g. "https://www.crystallography.net/cod/1534932.cif").
   */
  void read(const std::string &path) override;

  /**
   * @brief Returns the last successfully read Structure.
   *
   * @return A Structure with cell parameters and atomic coordinates in Cartesian.
   */
  Structure getStructure() const override;

private:
  /**
   * @brief Convenience method to check if a string is an HTTP/HTTPS URL.
   */
  bool isHttpUrl(const std::string &path) const;

  /**
   * @brief Extracts a COD material ID (e.g. "1534932") from a full COD URL.
   *        If parsing fails, returns an empty string.
   */
  std::string extractCodIdFromUrl(const std::string &url) const;

  /**
   * @brief Uses CODDataProvider to fetch raw CIF text by ID (or returns an error if not found).
   */
  std::string fetchCifById(const std::string &codID) const;

  /**
   * @brief Internal buffer for storing the last parsed structure.
   */
  Structure m_structure;

  /**
   * @brief COD data provider for HTTP requests. No parsing is performed in the provider.
   */
  std::unique_ptr<IMaterialsDataProvider> m_dataProvider;
};

} // namespace mlip
} // namespace tadah
#endif // COD_READER_H

