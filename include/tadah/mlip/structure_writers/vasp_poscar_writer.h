#ifndef VASP_POSCAR_WRITER_H
#define VASP_POSCAR_WRITER_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/structure_writer.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
namespace tadah {
namespace mlip {
/**
 * @brief Writes a structure as a VASP POSCAR: the rows of Structure::cell
 * as the lattice lines (scale 1.0) and Cartesian coordinates. VASP stops
 * on a left-handed basis, so such a cell is written as (a0, a1, -a2), the
 * same lattice right-handed, with the same Cartesian positions. A singular
 * cell is refused (std::runtime_error naming the structure) before the
 * file is opened.
 */
class VaspPoscarWriter : public StructureWriter {
public:
  VaspPoscarWriter(StructureDB& db);

  virtual void write_data(const std::string& filename, const size_t i) override;

  /** The index, and a singular cell (coplanar or zero lattice vectors). */
  virtual void check(const size_t i) const override;

};

} // namespace mlip
} // namespace tadah
#endif // VASP_POSCAR_WRITER_H
