#ifndef CIF_READER_H
#define CIF_READER_H

#include <tadah/mlip/structure_readers/istructure_reader.h>
#include <tadah/mlip/structure.h>
#include <string>
namespace tadah {
namespace mlip {

/**
 * @brief Reads Structure data from a CIF file using older Gemmi APIs.
 *
 * - Gets cell parameters from block.find_value("_cell_length_a"), etc.
 * - Iterates block.items to find loops (where item.kind == gemmi::cif::Item::Loop).
 * - Searches loop.tags for _atom_site_fract_x or _atom_site_cartn_x.
 * - Uses loop.get_s(row, col) to retrieve the data cells, converting to float/double.
 */
class CifReader : public IStructureReader {
public:
  CifReader() = default;

  void read(const std::string &path) override;
  Structure getStructure() const override;

  /**
   * @brief Parses the entire CIF text, populating m_structure with cell & atoms.
   */
  void parseCifContents(const std::string &contents);

private:
  Structure m_structure;

  /**
   * @brief Creates a 3×3 matrix from (a,b,c, alpha,beta,gamma).
   *        a along x, b in xy-plane, c forms a right-handed set.
   */
  void buildCellFromABC(double aVal, double bVal, double cVal,
                        double alphaDeg, double betaDeg, double gammaDeg);

  /**
   * @brief Convert (fx,fy,fz) fractional → Cartesian coords using m_structure.cell.
   */
  void fracToCart(double fx, double fy, double fz,
                  double &cx, double &cy, double &cz) const;

  /**
   * @brief Convert degrees to radians.
   */
  static double deg2rad(double deg);
};

} // namespace mlip
} // namespace tadah
#endif // CIF_READER_H

