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
   * @brief An AflowReader that fetches through @p provider (a stub in tests);
   *        a null provider is refused (std::invalid_argument).
   */
  explicit AflowReader(std::unique_ptr<IMaterialsDataProvider> provider);

  /**
   * @brief Reads structural data by an AUID string ("aflow:e9c6d914c4b8d9ca").
   *
   * A failed fetch throws std::runtime_error naming the HTTP status (or the
   * provider's own: 404 for an AFLUX answer without an entry, 500 for a
   * network error), the AUID and the provider's reason, cut to 300
   * characters; the previous structure is kept.
   */
  void read(const std::string &auid) override;

  /**
   * @brief Returns the last retrieved structure (cell + atoms).
   */
  Structure getStructure() const override;

  /**
   * @brief Reads geometry, species, composition and positions_fractional
   *        (and auid, compound, volume_cell and density for the label)
   *        from an aflowlib entry.
   *
   * Each field is read in either encoding, whatever the encoding of the
   * others: the string form ("geometry": "a,b,c,alpha,beta,gamma",
   * "species": "Cl,Na", "composition": "1,1", "positions_fractional":
   * "0,0,0;0.5,0.5,0.5", "volume_cell": "44.85154") or the typed form of
   * the entry file aflowlib.json (a six-number array, an array of symbols,
   * an array of counts, an array of three-number arrays, a number). auid
   * and compound are strings in both. A number in the label is written as
   * JSON writes it, so both forms give the same label.
   *
   * @throws std::runtime_error naming the entry's auid and the field (only
   *         the field when "auid" itself is not a string) for: an entry
   *         without geometry, species, composition or positions_fractional;
   *         a species that is not an element symbol Tadah knows ("AflowReader:
   *         entry '<auid>': species 'D' is not an element symbol Tadah knows
   *         (H to Og)."); a field in neither form; a geometry of other than six numbers; a
   *         site of other than three numbers; a composition count that is
   *         not a whole number 0 or more; and species, composition counts
   *         and positions that do not match. Also for a geometry that
   *         describes no cell (see makeCellMatrix) and for text that is not
   *         a JSON object. The structure of the last successful read is
   *         then kept.
   *
   * Public so that an AFLOW entry can be parsed without a network query
   * (as CifReader::parseCifContents does for CIF text).
   */
  void parseAflowJson(const std::string &jsonContent);

private:
  Structure m_structure;
  std::unique_ptr<IMaterialsDataProvider> m_dataProvider;

  /**
   * @brief Calls AFLOWDataProvider->QueryByID(auid), then parses the
   *        returned entry (aflowlib.json) with parseAflowJson.
   */
  void fetchAndParseAflowByAuid(const std::string &auid);

  /**
   * @brief Build a 3×3 cell from (a,b,c, α,β,γ) in degrees (a along x, b
   *        in the xy plane; rows are the lattice vectors). Throws when no
   *        cell has these lengths and angles.
   */
  void makeCellMatrix(double a, double b, double c,
                      double alpha, double beta, double gamma,
                      double (&cell)[3][3]);
};

} // namespace mlip
} // namespace tadah
#endif // AFLOW_READER_H
