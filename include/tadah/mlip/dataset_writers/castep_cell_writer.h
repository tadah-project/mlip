#ifndef CASTEP_CELL_WRITER_H
#define CASTEP_CELL_WRITER_H

#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_writers/dataset_writer.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
class CastepCellWriter : public DatasetWriter {
public:
  CastepCellWriter(StructureDB& db);

  virtual void write_data(const std::string& filename, const size_t i) override;

};

#endif // CASTEP_CELL_WRITER_H


