#ifndef MATERIALS_PROJECT_READER_H
#define MATERIALS_PROJECT_READER_H

#include <string>
#include <tadah/mlip/structure_readers/structure_reader.h>

/**
 * @brief Reads a Structure from the Materials Project via REST API (v2).
 *
 * Usage:
 * @code
 *   MaterialsProjectReader reader("MY_VALID_MP_KEY");
 *   reader.read("mp-149");
 *   Structure s = reader.getStructure();
 * @endcode
 */
class MaterialsProjectReader : public StructureReader {
public:
  explicit MaterialsProjectReader(const std::string &apiKey);

  void read(const std::string &mpID) override;
  Structure getStructure() const override;

private:
  std::string m_apiKey;
  Structure m_structure;

  void fetchAndParseMP(const std::string &mpID);
  std::string httpGet(const std::string &url);
  void parseMpJson(const std::string &jsonContent);
};

#endif // MATERIALS_PROJECT_READER_H
