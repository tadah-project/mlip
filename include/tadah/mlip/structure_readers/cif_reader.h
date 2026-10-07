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
 * A CIF holds one data block (one structure); an error gemmi raises while
 * parsing it, or reading the block (make_small_structure_from_block: the
 * cell parameters and the atom sites), is rethrown as std::runtime_error
 * naming the file (or the origin given to parseCifContents, or the text
 * and its data block), as every error and WARNING below does. The cell is
 * read by the reader: the three lengths must be given, an angle left out
 * (or '.') is 90 degrees, the CIF dictionary's default, and an unknown
 * length or angle ('?') is refused naming the tag (gemmi kept a 1 A cube unless all
 * six were given, and read a '.' or '?' angle among them as NaN). Of each pair of
 * symmetry tags (the operator loop _space_group_symop_operation_xyz or
 * _symmetry_equiv_pos_as_xyz, the H-M symbol _space_group_name_H-M_alt or
 * _symmetry_space_group_name_H-M, the Hall symbol _space_group_name_Hall
 * or _symmetry_space_group_name_Hall, the number _space_group_IT_number or
 * _symmetry_Int_Tables_number) the first with a known value is used: an
 * unknown ('?', '.') or blank one does not hide the other tag. The H-M
 * symbol is first put in the form gemmi's table lookup reads: blanks
 * trimmed, '_' removed ('P 2_1/c'), subscripts in parentheses closed up
 * ('P 4(2)/n m c'), a trailing ' (No. N)' dropped, the CIF dictionary's
 * notes '(origin choice N)', '(origin at -1)' (origin choice 2, or the
 * group's only origin), '(hexagonal axes)' and '(rhombohedral axes)' read
 * as ':N', ':2', ':H' and ':R', ICSD's ' S' and ' Z' suffixes (either
 * case) as origin choice 1 and 2, and an R group's axes written as a
 * suffix ('R -3 m H', 'R3R') as ':H' / ':R'. A full symbol, which the
 * dictionary asks for and FINDSYM writes ('I 4/m -3 2/m', 'P 63/m 2/m
 * 2/c', 'R -3 2/m'), is read by its short one (ITA's rule: every 'n/x'
 * reduced to 'x' for an orthorhombic or cubic group, all but the first
 * for a tetragonal, trigonal or hexagonal one). A value that this leaves
 * empty ('(No. 14)', '_') cannot be read.
 *
 * The space group comes from the symmetry-operator loop
 * (_space_group_symop_operation_xyz or _symmetry_equiv_pos_as_xyz) when
 * there is one: a tabulated setting, or the loop's own operators when they
 * form a group in no tabulated setting (the operators listed, closed
 * under products, translations included). Without a loop, or when the loop forms no
 * group, it comes from the Hall symbol (one in no tabulated setting
 * expands the sites with its own operators), else the H-M symbol, each in
 * the setting it names or else in the standard one: hexagonal or
 * rhombohedral axes from the cell angles, origin choice 2 where the group
 * has two (also for the current symbols gemmi lists apart, such as
 * 'C c c e' for C c c a); else the space-group number, in its standard
 * setting. A symbol used so that cannot be read (or an H-M symbol that is
 * a number) is refused, quoting it ("cannot read"); the Hall symbol is
 * read before the H-M symbol, and an unreadable one is refused even when
 * the H-M symbol is fine. Next to a loop that forms a group, or a readable
 * Hall symbol, the symbols are only compared with it: one that cannot be
 * read, or is a number, gives the WARNING below. The number is read only
 * when it is used (quoted or not), and must be an integer from 1 to 230
 * (0 or '229.0' is refused there); when the cell could be in another ITA
 * setting of its group (another origin choice, or another choice of axes
 * that fits the cell) a WARNING names the one used. A loop that forms no
 * group also gives the WARNING below, which says so and quotes gemmi's
 * diagnostic of the loop.
 * A CIF with no symmetry record is P 1.
 *
 * However it was given, a monoclinic group must fit the cell: the angles
 * between its unique axis and the other two axes (alpha and gamma for
 * unique axis b) are 90 degrees within 1e-3, or the CIF is refused, naming
 * the group, the record it came from and the angles. So must a group with
 * a 3-fold rotation about c (every trigonal and hexagonal group on
 * hexagonal axes, R ':H' or P) alpha = beta = 90 and gamma = 120, and an
 * R group on rhombohedral axes (':R') alpha = beta = gamma, within 1e-3
 * degrees (an R symbol without its axes takes them from the cell). The
 * axes come from the rotations alone: translations (a centring, a
 * supercell's x+1/3) fix nothing. A group in no tabulated setting (an
 * origin moved, an unusual centring) is checked as the tabulated group
 * with its rotations and centring, else by a 3-fold rotation about c (R in
 * the reverse setting), else as the tabulated primitive group with its
 * rotations; one whose rotations are those of no tabulated setting (a
 * monoclinic group on the axes of a primitive cell of a centred lattice,
 * say) is not checked. A refusal advises another setting only when one
 * fits the cell, and otherwise correcting the cell. A short symbol
 * is read in the setting gemmi gives it: unique axis b, except the
 * B-centred 'B 2', 'B m', 'B b', 'B n', 'B 2/m', 'B 2/b' and 'B 2/n'
 * (unique axis c, as in the PDB; 'B 21' is unique axis b); a number means
 * unique axis b.
 *
 * Every site of the unit cell (get_all_unit_cell_sites) becomes an atom at
 * its Cartesian position; gemmi merges the images of one site within
 * 0.4 A, never two _atom_site rows.
 *
 * When the symmetry record is inconsistent (gemmi's
 * SmallStructure::check_spacegroup() for the loop, the Hall symbol and
 * the number; an H-M symbol that names another group than the one the
 * cell is built with, or another setting of it, such as another origin
 * choice, compared by the operations it names, so an equivalent name of
 * the same setting such as 'C c c b:1' agrees, and a name without a
 * setting ('F d -3 m', 'C c c b') agrees with any setting of that name;
 * next to a loop or a readable Hall symbol, an H-M symbol that is a
 * number or cannot be read; a number that is not an integer and is not
 * used), a WARNING on std::cerr names the file (or the origin, or the
 * text and its data block), quotes the findings and says what the unit
 * cell is built with.
 *
 * A site whose _atom_site_occupancy differs from 1 by more than 1e-6 is
 * refused with std::runtime_error, "site <label> has occupancy <occ>" (as
 * written):
 * Tadah has no representation for a disordered or partly vacant site, and
 * an occupancy above 1 is not valid. An unknown occupancy ('?' or '.') is
 * 1; one that is not a number is refused.
 *
 * A site of an element Tadah does not know is refused, naming the site:
 * deuterium, which gemmi reads as an element ("site D1 has element 'D',
 * which Tadah does not know: its elements are H to Og."), or a type symbol
 * (or, without the type-symbol column, a label) that gemmi cannot read as
 * an element ("site Q1: 'Q' is not an element symbol.").
 *
 * The cell is gemmi's orthogonalisation (a along x, b in the xy plane),
 * whose columns are the lattice vectors; it is stored transposed, so row
 * i of Structure::cell is lattice vector a_i.
 *
 * The structure's label names where the text came from, the space group
 * the unit cell is built with and the cell volume (cubic angstrom, 6
 * decimals), e.g. for read(), parseCifContents(text) and
 * parseCifContents(text, origin) (CODReader's origin):
 *
 *     CIF file: si.cif | Symmetry: F d -3 m:2 | Volume: 160.191478
 *     CIF text (data_Si) | Symmetry: F d -3 m:2 | Volume: 160.191478
 *     COD entry 1534932 | Symmetry: I m -3 m | Volume: 23.553521
 *
 * The group is named as the WARNINGs name it: the tabulated setting
 * (gemmi's extended H-M symbol: 'F d -3 m:2', 'R -3 m:H', 'I 1'), the
 * number of operators and the record for a group in no tabulated setting
 * ("2 operators of the symmetry-operator loop"), or 'P 1' without a
 * symmetry record. The H-M symbol as written appears in the WARNINGs,
 * where it differs.
 *
 * A read replaces the reader's structure only when it succeeds: a CIF that
 * is refused leaves the last structure (cell, atoms and label) as it was.
 */
class CifReader : public IStructureReader {
public:
  CifReader() = default;

  void read(const std::string &path) override;
  Structure getStructure() const override;

  /**
   * @brief Parses the entire CIF text, populating m_structure with cell & atoms.
   *
   * There is no file: warnings, errors and the label name the data block
   * ("CIF text (data_<name>)"), or, before one is found (a syntax error,
   * two data blocks), the text ("CIF text"). The overload below names an
   * origin instead.
   */
  void parseCifContents(const std::string &contents);

  /**
   * @brief Parses CIF text that came from @p origin, e.g. "COD entry
   *        1534932" (CODReader).
   *
   * A non-empty @p origin names the text in every error and WARNING
   * ("COD entry 1534932: site Fe1 has occupancy 0.5; ...") and starts the
   * label; the data block is not named. An empty one is the one-argument
   * form.
   */
  void parseCifContents(const std::string &contents,
                        const std::string &origin);

private:
  Structure m_structure;

  /**
   * @brief Parses CIF text. @p where names it in errors and WARNINGs
   *        ("CIF file <path>", "CIF text", an origin); with @p name_block
   *        the data block is added once it is known ("CIF text
   *        (data_<name>)"). The label starts with @p label_origin, or with
   *        that name when @p label_origin is empty. m_structure is replaced
   *        only when the whole structure is built.
   */
  void parseCif(const std::string &contents, const std::string &where,
                bool name_block, const std::string &label_origin);
};

} // namespace mlip
} // namespace tadah
#endif // CIF_READER_H

