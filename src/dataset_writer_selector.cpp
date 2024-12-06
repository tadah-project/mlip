#include <tadah/mlip/dataset_writers/dataset_writer_selector.h>
#include <tadah/mlip/dataset_writers/castep_cell_writer.h>
#include <tadah/mlip/dataset_writers/vasp_poscar_writer.h>
#include <tadah/mlip/dataset_writers/lammps_structure_writer.h>

#include <fstream>
#include <iostream>

// Factory method implementation
std::unique_ptr<DatasetWriter> DatasetWriterSelector::get_writer(const std::string& type, StructureDB& db) {

  if (type == "CASTEP") {
    return std::make_unique<CastepCellWriter>(db);
  } else if (type == "VASP") {
    return std::make_unique<VaspPoscarWriter>(db);
  } else if (type == "LAMMPS") {
    return std::make_unique<LammpsStructureWriter>(db);
  } else {
    std::cerr << "Unknown type! Returning nullptr." << std::endl;
    return nullptr;
  }

}
