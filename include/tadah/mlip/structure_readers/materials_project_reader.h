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
   * @brief Parses MP JSON to retrieve the lattice matrix and atomic positions.
   */
  void parseMpJson(const std::string &jsonContent);
};

} // namespace mlip
} // namespace tadah
#endif // MATERIALS_PROJECT_READER_H
