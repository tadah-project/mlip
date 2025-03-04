#ifndef CASTEP_CELL_READER_H
#define CASTEP_CELL_READER_H

#include <string>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/structure_reader.h>

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
 * If positions are fractional, we multiply by the 3×3 cell matrix
 * to get absolute positions in Angstrom. If positions are absolute,
 * we store them directly.
 *
 * This parser also extracts atomic species from the first token
 * in POSITIONS_* blocks.
 */
class CastepCellReader : public StructureReader {
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
   * @brief Parses the complete contents of a .cell file from a single string.
   */
  void parseCellContents(const std::string &contents);

  /**
   * @brief Parses LATTICE_ABC lines into the 3×3 matrix in m_structure.cell.
   * @param lines a vector of strings containing a b c on one line, alpha beta
   * gamma on next line
   */
  void parseLatticeABC(const std::vector<std::string> &lines);

  /**
   * @brief Converts atomic fractional coordinates (fx, fy, fz) to absolute in Å
   *        using the current m_structure.cell matrix.
   */
  void applyFractional(double fx, double fy, double fz, double &cx, double &cy,
                       double &cz);

  /**
   * @brief Helper for mapping string species (e.g., "Si", "O") to Element enum.
   */
  Element parseSpecies(const std::string &sym);
};

#endif // CASTEP_CELL_READER_H
