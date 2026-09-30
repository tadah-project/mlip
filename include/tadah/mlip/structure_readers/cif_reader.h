#ifndef CIF_READER_H
#define CIF_READER_H

#include <tadah/mlip/structure_readers/istructure_reader.h>
#include <tadah/mlip/structure.h>
#include <string>
namespace tadah {
namespace mlip {

/**
 * @brief Reads a Structure from a CIF file with gemmi.
 *
 * gemmi::make_small_structure_from_block reads the cell parameters, the
 * symmetry operations and the atom sites; every site of the unit cell
 * (get_all_unit_cell_sites) becomes an atom at its Cartesian position.
 * The cell is gemmi's orthogonalisation (a along x, b in the xy plane),
 * whose columns are the lattice vectors; it is stored transposed, so row
 * i of Structure::cell is lattice vector a_i.
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
};

} // namespace mlip
} // namespace tadah
#endif // CIF_READER_H

