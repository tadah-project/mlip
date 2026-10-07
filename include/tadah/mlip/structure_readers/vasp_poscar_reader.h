#ifndef VASP_POSCAR_READER_H
#define VASP_POSCAR_READER_H

#include <tadah/mlip/structure_readers/istructure_reader.h>
namespace tadah {
namespace mlip {

/**
 * @brief Reads a VASP POSCAR/CONTCAR file.
 *
 * The format, line by line:
 *   1) Comment line
 *   2) Scaling factor: one number s > 0 (scales the lattice), s < 0 (the
 *      cell volume), or three numbers > 0 scaling the x, y and z
 *      components of the lattice vectors; anything else is refused
 *   3-5) Lattice vectors a0, a1, a2, stored as rows of Structure::cell
 *   6) Element symbols (required: a file without them, the VASP 4
 *      layout, is refused; a line 6 of atom counts is that layout)
 *   7) Atom counts per element, one per symbol: whole numbers written
 *      with digits, after at most one '+' ("2", "+2"); a '-', a fraction or
 *      trailing text is refused, as is a count past the int range
 *   8) Optional "Selective dynamics" line (first character S or s)
 *   9) Coordinate mode: first character C, c, K or k for Cartesian,
 *      anything else for Direct (fractional)
 *  10+) Atomic coordinates: fractional f give r = f0 a0 + f1 a1 + f2 a2;
 *      Cartesian ones are scaled like the lattice
 *
 * Every error of read() is a std::runtime_error naming the file:
 *
 *     VaspPoscarReader: POSCAR file <path>: 'Xx' on line 6 is not an element symbol (or a VASP POTCAR name such as Ti_pv or Si/b0d4a2).
 *
 * and a refused file leaves the structure of the last successful read.
 */
class VaspPoscarReader : public IStructureReader {
public:
  VaspPoscarReader() = default;

  void read(const std::string &path) override;
  Structure getStructure() const override;

private:
  Structure m_structure;
  /// The structure of a POSCAR's lines, without its label; its errors say
  /// what is wrong, and read() adds the file.
  static Structure parsePoscar(const std::vector<std::string> &lines);
};

} // namespace mlip
} // namespace tadah
#endif // VASP_POSCAR_READER_H
