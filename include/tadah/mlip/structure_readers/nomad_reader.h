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
   * @brief A NomadReader that fetches through @p provider (a stub in tests);
   *        a null provider is refused (std::invalid_argument).
   */
  explicit NomadReader(std::unique_ptr<IMaterialsDataProvider> provider);

  /**
   * @brief Reads structural data for the NOMAD entry ID. Example: "zyXabc123..."
   *
   * A failed fetch throws std::runtime_error naming the HTTP status (or the
   * provider's own: 500 for a cURL error such as no network), the entry ID
   * and the provider's reason, cut to 300 characters (failed_fetch_message()):
   *
   *     NOMAD: HTTP 500 for entry 'zyXabc123zyXabc123zyXab': NOMAD: cURL error: ...
   *
   * The reason is the provider's cURL text or the server's body; the
   * message holds no request URL or header. The previous structure is kept.
   *
   * An entry that cannot be read is refused naming it, the site of a
   * species Tadah does not know too:
   *
   *     NOMAD entry 'zyXabc123zyXabc123zyXab': site 2 has species 'D', which is not an element symbol Tadah knows (H to Og).
   *
   * and the previous structure is kept.
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
   * @brief The structure of the raw JSON response of entry @p entryID: cell
   *        parameters, positions, etc.
   *
   * Every error names the entry, "NOMAD entry '<entryID>': <what>":
   * a missing field, text that is not JSON, and a site whose species is not
   * an element symbol Tadah knows ("site 2 has species 'D', which is not an
   * element symbol Tadah knows (H to Og).").
   */
  static Structure parseNomadJson(const std::string &jsonContent,
                                  const std::string &entryID);

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

