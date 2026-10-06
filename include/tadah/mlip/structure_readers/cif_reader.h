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
 * symmetry record and the atom sites. The space group comes from the
 * symmetry-operator loop (_space_group_symop_operation_xyz or
 * _symmetry_equiv_pos_as_xyz) when there is one; gemmi falls back to the
 * Hall and H-M symbols when the loop does not form a group (and a CIF
 * with only a number is then P 1, with the WARNING below). Without a loop
 * it comes from the Hall symbol, the H-M symbol (_space_group_name_H-M_alt
 * or _symmetry_space_group_name_H-M) or the space-group number
 * (_space_group_IT_number or _symmetry_Int_Tables_number), in that order,
 * in the standard setting (hexagonal or rhombohedral axes from the cell
 * angles, origin choice 2 where the group has two; ICSD's ' S' and ' Z'
 * suffixes of the H-M symbol name origin choice 1 and 2). A symbol that
 * names no known group is refused; a number is used only without a
 * symbol, a monoclinic one only for a cell with unique axis b (alpha =
 * gamma = 90), and a number that has more than one setting gives a
 * WARNING naming the one used. A CIF with no symmetry record is P 1. Every site of the unit cell
 * (get_all_unit_cell_sites) becomes an atom at its Cartesian position;
 * gemmi merges the images of one site within 0.4 A, never two _atom_site
 * rows.
 *
 * When gemmi's SmallStructure::check_spacegroup() reports the symmetry
 * record inconsistent (say an H-M symbol that differs from the loop), a
 * WARNING on std::cerr names the file, quotes the diagnostic and says
 * what the unit cell is built with.
 *
 * A site whose _atom_site_occupancy differs from 1 by more than 1e-6 is
 * refused with std::runtime_error, "site <label> has occupancy <occ>" (as
 * written):
 * Tadah has no representation for a disordered or partly vacant site, and
 * an occupancy above 1 is not valid. An unknown occupancy ('?' or '.') is
 * 1; one that is not a number is refused.
 *
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
   *
   * There is no file: warnings and errors name the data block instead.
   */
  void parseCifContents(const std::string &contents);

private:
  Structure m_structure;

  /**
   * @brief Parses CIF text read from @p path (empty when there is no file),
   *        which warnings and errors name.
   */
  void parseCif(const std::string &contents, const std::string &path);
};

} // namespace mlip
} // namespace tadah
#endif // CIF_READER_H

