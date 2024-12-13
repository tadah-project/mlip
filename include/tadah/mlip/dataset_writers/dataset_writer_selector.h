#ifndef DATASET_WRITER_SELECTOR_H
#define DATASET_WRITER_SELECTOR_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_writers/dataset_writer.h>
#include <string>
#include <memory>

class DatasetWriterSelector {
  public:
    static std::unique_ptr<DatasetWriter> get_writer(const std::string& type, StructureDB& db);

};

#endif // DATASET_WRITER_SELECTOR_H
