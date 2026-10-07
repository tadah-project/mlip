#ifndef CASTEP_CELL_READER_H
#define CASTEP_CELL_READER_H

#include <string>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/istructure_reader.h>
namespace tadah {
namespace mlip {

/**
 * @brief Reads crystallographic and atomic data from a CASTEP .cell file.
 *
 * The .cell file can contain:
 *  - %BLOCK LATTICE_CART  or %BLOCK LATTICE_ABC
 *  - %BLOCK POSITIONS_FRAC or %BLOCK POSITIONS_ABS
 *
 * If LATTICE_ABC is provided, the code converts (a,b,c,α,β,γ) into
 * a 3×3 Cartesian representation with:
 *   a along x,
 *   b in xy plane,
 *   c forming a right-handed set with a and b.
 *
 * Each LATTICE_CART line is a lattice vector, stored as a row of
 * Structure::cell. Fractional positions f become r = f0 a + f1 b + f2 c
 * (cell^T f), converted after the whole file is read, so the blocks may
 * come in any order. An optional first line naming a length unit (ang,
 * bohr, a0, nm, cm, m, pm; default ang) is applied in LATTICE_CART,
 * LATTICE_ABC and POSITIONS_ABS. Text after '!', '#' or ';' on any line
 * is a comment. A file without a lattice block or a positions block, a
 * LATTICE_CART without exactly three vectors, an incomplete LATTICE_ABC,
 * a singular lattice, an unknown unit or a positions block without atoms
 * is refused (std::runtime_error), and so is a species label that does not
 * start with an element symbol. Every error of read() names the file:
 *
 *     CastepCellReader: CASTEP cell file <path>: CASTEP species label 'Qq:up' does not start with an element symbol
 *
 * and a refused file leaves the structure of the last successful read.
 *
 * This parser also extracts atomic species from the first token
 * in POSITIONS_* blocks.
 */
class CastepCellReader : public IStructureReader {
public:
  /**
   * @brief Default constructor.
   */
  CastepCellReader() = default;

  /**
   * @brief Reads and parses the specified .cell file.
   * @param path Path to the CASTEP cell file.
   */
  void read(const std::string &path) override;

  /**
   * @brief Returns the parsed Structure with absolute coordinates in Ångström.
   */
  Structure getStructure() const override;

private:
  Structure m_structure;

  /**
   * @brief The structure of the complete contents of a .cell file, without
   *        its label; its errors say what is wrong, and read() adds the
   *        file.
   */
  static Structure parseCellContents(const std::string &contents);

  /**
   * @brief Parses LATTICE_ABC lines into the 3×3 matrix in st.cell.
   * @param lines two lines: a b c, then alpha beta gamma (degrees); the
   *        block's unit line is already removed
   * @param unit factor converting the block's length unit to Angstrom
   * @param st the structure whose cell is set
   */
  static void parseLatticeABC(const std::vector<std::string> &lines,
                              double unit, Structure &st);

  /**
   * @brief Converts atomic fractional coordinates (fx, fy, fz) to absolute in Å
   *        using the cell matrix of @p st.
   */
  static void fracToAbs(const Structure &st, double fx, double fy, double fz,
                        double &cx, double &cy, double &cz);

  /**
   * @brief Helper for mapping string species (e.g., "Si", "O") to tadah::core::Element enum.
   */
  static tadah::core::Element parseSpecies(const std::string &sym);
};

} // namespace mlip
} // namespace tadah
#endif // CASTEP_CELL_READER_H
