#include <tadah/mlip/structure_writers/structure_writer_selector.h>
#include <tadah/mlip/structure_writers/castep_cell_writer.h>
#include <tadah/mlip/structure_writers/vasp_poscar_writer.h>
#include <tadah/mlip/structure_writers/lammps_structure_writer.h>

#include <stdexcept>
namespace tadah {
namespace mlip {

// Factory method implementation
std::unique_ptr<StructureWriter> StructureWriterSelector::get_writer(const std::string& type, StructureDB& db) {

  if (type == "CASTEP") {
    return std::make_unique<CastepCellWriter>(db);
  } else if (type == "VASP") {
    return std::make_unique<VaspPoscarWriter>(db);
  } else if (type == "LAMMPS") {
    return std::make_unique<LammpsStructureWriter>(db);
  }
  throw std::invalid_argument("unknown structure format '" + type +
                              "': the formats written are CASTEP, LAMMPS and "
                              "VASP");

}
} // namespace mlip
} // namespace tadah
