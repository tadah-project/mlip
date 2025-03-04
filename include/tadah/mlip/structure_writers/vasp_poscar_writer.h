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
class VaspPoscarWriter : public StructureWriter {
public:
  VaspPoscarWriter(StructureDB& db);

  virtual void write_data(const std::string& filename, const size_t i) override;

};

#endif // VASP_POSCAR_WRITER_H
