#ifndef LAMMPS_STRUCTURE_WRITER_H
#define LAMMPS_STRUCTURE_WRITER_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_writers/dataset_writer.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
class LammpsStructureWriter : public DatasetWriter {
public:
  LammpsStructureWriter(StructureDB& db);

  virtual void write_data(const std::string& filename, const size_t i) override;

};

#endif // LAMMPS_STRUCTURE_WRITER_H
