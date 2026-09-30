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
 *      layout, is refused)
 *   7) Atom counts per element, one per symbol
 *   8) Optional "Selective dynamics" line (first character S or s)
 *   9) Coordinate mode: first character C, c, K or k for Cartesian,
 *      anything else for Direct (fractional)
 *  10+) Atomic coordinates: fractional f give r = f0 a0 + f1 a1 + f2 a2;
 *      Cartesian ones are scaled like the lattice
 */
class VaspPoscarReader : public IStructureReader {
public:
  VaspPoscarReader() = default;

  void read(const std::string &path) override;
  Structure getStructure() const override;

private:
  Structure m_structure;
  void parsePoscar(const std::vector<std::string> &lines);
};

} // namespace mlip
} // namespace tadah
#endif // VASP_POSCAR_READER_H
