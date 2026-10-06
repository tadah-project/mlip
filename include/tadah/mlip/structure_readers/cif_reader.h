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
 * symmetry record and the atom sites. Of each pair of symmetry tags (the
 * H-M symbol _space_group_name_H-M_alt or _symmetry_space_group_name_H-M,
 * the Hall symbol _space_group_name_Hall or
 * _symmetry_space_group_name_Hall, the number _space_group_IT_number or
 * _symmetry_Int_Tables_number) the first known value is used: an unknown
 * one ('?' or '.') does not hide the other tag. The H-M symbol is first
 * put in the form gemmi's table lookup reads: blanks trimmed, '_' removed
 * ('P 2_1/c'), subscripts in parentheses closed up ('P 4(2)/n m c'), a
 * trailing ' (No. N)' dropped, and ICSD's ' S' and ' Z' suffixes (either
 * case) read as origin choice 1 and 2. A value of blanks is no symbol; one
 * that this otherwise leaves empty ('(No. 14)', '_') cannot be read.
 *
 * The space group comes from the symmetry-operator loop
 * (_space_group_symop_operation_xyz or _symmetry_equiv_pos_as_xyz) when
 * there is one: a tabulated setting, or the loop's own operators when they
 * form a group in no tabulated setting. Without a loop, or when the loop
 * forms no group, it comes from the Hall symbol (one in no tabulated
 * setting expands the sites with its own operators), else the H-M symbol,
 * each in the setting it names or else in the standard one: hexagonal or
 * rhombohedral axes from the cell angles, origin choice 2 where the group
 * has two (also for the current symbols gemmi lists apart, such as
 * 'C c c e' for C c c a). A symbol that cannot be read (or an H-M symbol
 * that is a number) is refused, quoting it ("cannot read"); the Hall
 * symbol is read before the H-M symbol, and an unreadable one is refused
 * even when the H-M symbol is fine. Without a loop and without a symbol it
 * comes from the space-group number, which must be 1 to 230 (0 is
 * refused), in its standard setting; when the cell could be in another
 * ITA setting of the group (another origin choice, or another choice of
 * axes that fits the cell) a WARNING names the one used. With a loop that
 * forms no group and no symbol, the number is not used: the cell is P 1,
 * with the WARNING below quoting gemmi's diagnostic of the loop. A CIF
 * with no symmetry record is P 1.
 *
 * However it was given, a monoclinic group must fit the cell: the angles
 * between its unique axis and the other two axes (alpha and gamma for
 * unique axis b) are 90 degrees within 1e-3, or the CIF is refused, naming
 * the group, the record it came from and the angles. A group in no
 * tabulated setting (an origin moved, an unusual centring) is checked as
 * the tabulated primitive group with its rotations; one whose rotations
 * are those of no tabulated setting (a monoclinic group on the axes of a
 * primitive cell of a centred lattice, say) is not checked. A short symbol
 * is read in the setting gemmi gives it: unique axis b, except 'B 2',
 * 'B m', 'B b', 'B 2/m' and 'B 2/b' (unique axis c, as in the PDB); a
 * number means unique axis b.
 *
 * Every site of the unit cell (get_all_unit_cell_sites) becomes an atom at
 * its Cartesian position; gemmi merges the images of one site within
 * 0.4 A, never two _atom_site rows.
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

