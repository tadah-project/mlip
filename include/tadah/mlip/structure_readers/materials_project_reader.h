#ifndef MATERIALS_PROJECT_READER_H
#define MATERIALS_PROJECT_READER_H

#include <memory>
#include <string>

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <tadah/mlip/data_providers/materials_project_data_provider.h>
#include <tadah/mlip/structure_readers/istructure_reader.h>
#include <tadah/mlip/structure.h>
namespace tadah {
namespace mlip {

/**
 * @brief Reads a Structure from the Materials Project via REST API.
 *        Uses MaterialsProjectDataProvider to retrieve raw JSON,
 *        then parses it here.
 */
class MaterialsProjectReader : public IStructureReader {
public:
  /**
   * @brief Constructor storing the Materials Project API key.
   *
   * @param apiKey  Must be valid for the MP service (e.g. "ABCDEFG12345").
   */
  explicit MaterialsProjectReader(const std::string &apiKey);

  /**
   * @brief A MaterialsProjectReader that fetches through @p provider (a stub
   *        in tests); a null provider is refused (std::invalid_argument).
   */
  explicit MaterialsProjectReader(
      std::unique_ptr<IMaterialsDataProvider> provider);

  /**
   * @brief Reads structural data for e.g. "mp-149".
   *
   * A failed fetch throws std::runtime_error naming the HTTP status (or the
   * provider's own: 500 for a cURL error such as no network), the material
   * ID and the provider's reason, cut to 300 characters
   * (failed_fetch_message()):
   *
   *     Materials Project: HTTP 500 for material 'mp-149': Materials Project: cURL error: ...
   *
   * The reason is the provider's cURL text or the server's body; the
   * message holds no request URL, no header and not the API key (the
   * provider sends the key in a request header). The previous structure is
   * kept.
   *
   * A material that cannot be read is refused naming it, the site of an
   * element Tadah does not know too:
   *
   *     Materials Project material 'mp-149': site 2 has element 'Xx', which is not an element symbol Tadah knows (H to Og).
   *
   * and the previous structure is kept.
   */
  void read(const std::string &mpID) override;

  /**
   * @brief Returns the last parsed structure.
   */
  Structure getStructure() const override;

private:
  /// The final parsed structure after reading from MP.
  Structure m_structure;
  /// The data provider that fetches raw JSON from MP.
  std::unique_ptr<IMaterialsDataProvider> m_dataProvider;

  /**
   * @brief Calls MaterialsProjectDataProvider->QueryByID(mpID), then parse the result.
   */
  void fetchAndParseMP(const std::string &mpID);

  /**
   * @brief The structure of material @p mpID's MP JSON: the lattice matrix
   *        and the atomic positions.
   *
   * Every error names the material, "Materials Project material
   * '<mpID>': <what>": a missing field, text that is not JSON, and a site
   * whose element is not an element symbol Tadah knows ("site 2 has element
   * 'Xx', which is not an element symbol Tadah knows (H to Og).").
   */
  static Structure parseMpJson(const std::string &jsonContent,
                               const std::string &mpID);
};

} // namespace mlip
} // namespace tadah
#endif // MATERIALS_PROJECT_READER_H
