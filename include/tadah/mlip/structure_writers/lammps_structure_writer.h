#ifndef LAMMPS_STRUCTURE_WRITER_H
#define LAMMPS_STRUCTURE_WRITER_H

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
 * @brief Writes a structure as a LAMMPS data file (atom_style atomic).
 *
 * The box uses the general triclinic header (avec, bvec, cvec and abc
 * origin, which only LAMMPS versions with general triclinic boxes read):
 * avec, bvec and cvec are the rows of Structure::cell, and LAMMPS rotates
 * the box and the atoms into its own restricted orientation when it reads
 * the file. A left-handed cell is
 * written as (a0, a1, -a2) with "abc origin" a2: a right-handed basis of
 * the same lattice spanning the same parallelepiped, so every atom inside
 * the cell stays inside the box and each lattice direction keeps its
 * boundary flag. A singular cell is refused (std::runtime_error naming
 * the structure) before the file is opened.
 */
class LammpsStructureWriter : public StructureWriter {
public:
  LammpsStructureWriter(StructureDB& db);

  virtual void write_data(const std::string& filename, const size_t i) override;

};

} // namespace mlip
} // namespace tadah
#endif // LAMMPS_STRUCTURE_WRITER_H
