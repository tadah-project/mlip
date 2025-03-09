#ifndef VASP_POSCAR_READER_H
#define VASP_POSCAR_READER_H

#include <tadah/mlip/structure_readers/istructure_reader.h>
namespace tadah {
namespace mlip {

/**
 * @brief Reads a VASP POSCAR/CONTCAR file.
 *
 * A typical format:
 *   1) Comment line
 *   2) Scaling factor
 *   3-5) Lattice vectors
 *   6) tadah::core::Element symbols or counts
 *   7) Possibly "Selective dynamics" line
 *   8) "Direct" or "Cartesian"
 *   9+) Atomic coordinates
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
