#ifndef NOMAD_READER_H
#define NOMAD_READER_H

#include <memory>
#include <string>

#include <tadah/mlip/data_providers/imaterials_data_provider.h>
#include <tadah/mlip/data_providers/nomad_data_provider.h>
#include <tadah/mlip/structure_readers/istructure_reader.h>
#include <tadah/mlip/structure.h>
namespace tadah {
namespace mlip {

/**
 * @class NomadReader
 * @brief Fetches structural data from NOMAD by entry ID, then parses it into a Structure.
 *
 * Networking is delegated to NomadDataProvider. The raw JSON is parsed locally to extract
 * lattice, atomic positions, density, etc.
 */
class NomadReader : public IStructureReader {
public:
  /**
   * @brief Default constructor. Uses NomadDataProvider for network calls.
   */
  NomadReader();

  /**
   * @brief Reads structural data for the NOMAD entry ID. Example: "zyXabc123..."
   *
   * @param entryID  The unique identifier in NOMAD.
   */
  void read(const std::string &entryID) override;

  /**
   * @brief Returns the last retrieved structure from NOMAD.
   */
  Structure getStructure() const override;

private:
  /// The last successfully parsed structure.
  Structure m_structure;

  /// Data provider that fetches raw JSON from NOMAD (no parsing).
  std::unique_ptr<IMaterialsDataProvider> m_dataProvider;

  /**
   * @brief Calls NomadDataProvider->QueryByID(entryID), then parses the JSON to fill m_structure.
   */
  void fetchAndParseNomad(const std::string &entryID);

  /**
   * @brief Interprets the raw JSON response to extract cell parameters, positions, etc.
   */
  void parseNomadJson(const std::string &jsonContent);

  /**
   * @brief Builds a 3×3 lattice from edges (a,b,c) and angles (α,β,γ) in radians.
   */
  void makeCellMatrix(double a, double b, double c,
                      double alpha, double beta, double gamma,
                      double (&cell)[3][3]);
};

} // namespace mlip
} // namespace tadah
#endif // NOMAD_READER_H

