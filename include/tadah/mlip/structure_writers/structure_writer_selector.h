#ifndef STRUCTURE_WRITER_SELECTOR_H
#define STRUCTURE_WRITER_SELECTOR_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/structure_writer.h>
#include <string>
#include <memory>

class StructureWriterSelector {
  public:
    static std::unique_ptr<StructureWriter> get_writer(const std::string& type, StructureDB& db);

};

#endif // STRUCTURE_WRITER_SELECTOR_H
