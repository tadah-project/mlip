#ifndef AFLOW_READER_H
#define AFLOW_READER_H

#include <memory>
#include <string>

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <tadah/mlip/data_providers/aflow_data_provider.h>
#include <tadah/mlip/structure_readers/istructure_reader.h>
#include <tadah/mlip/structure.h>
namespace tadah {
namespace mlip {

/**
 * @class AflowReader
 * @brief Uses AFLOWDataProvider to fetch raw JSON via an AUID,
 *        then parses it to build a Structure object.
 */
class AflowReader : public IStructureReader {
public:
  AflowReader();

  /**
   * @brief Reads structural data by an AUID string ("aflow:e9c6d914c4b8d9ca").
   */
  void read(const std::string &auid) override;

  /**
   * @brief Returns the last retrieved structure (cell + atoms).
   */
  Structure getStructure() const override;

  /**
   * @brief Reads geometry, positions_cartesian, stoichiometry, etc. from AFLOW JSON.
   *
   * Public so that an AFLOW entry can be parsed without a network query
   * (as CifReader::parseCifContents does for CIF text).
   */
  void parseAflowJson(const std::string &jsonContent);

private:
  Structure m_structure;
  std::unique_ptr<IMaterialsDataProvider> m_dataProvider;

  /**
   * @brief Calls AFLOWDataProvider->QueryByID(auid), then parse the returned JSON.
   */
  void fetchAndParseAflowByAuid(const std::string &auid);

  /**
   * @brief Build a 3×3 cell from (a,b,c, α,β,γ) in degrees.
   */
  void makeCellMatrix(double a, double b, double c,
                      double alpha, double beta, double gamma,
                      double (&cell)[3][3]);
};

} // namespace mlip
} // namespace tadah
#endif // AFLOW_READER_H
