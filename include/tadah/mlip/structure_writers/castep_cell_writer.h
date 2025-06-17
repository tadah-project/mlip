#ifndef CASTEP_CELL_WRITER_H
#define CASTEP_CELL_WRITER_H

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
class CastepCellWriter : public StructureWriter {
public:
  CastepCellWriter(StructureDB& db);

  virtual void write_data(const std::string& filename, const size_t i) override;

};

} // namespace mlip
} // namespace tadah
#endif // CASTEP_CELL_WRITER_H


