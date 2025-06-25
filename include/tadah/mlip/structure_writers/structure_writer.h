#ifndef STRUCTURE_WRITER_H
#define STRUCTURE_WRITER_H

#include <tadah/mlip/structure_db.h>
#include <string>
#include <vector>
namespace tadah {
namespace mlip {

class StructureWriter {
  public:
    virtual ~StructureWriter() = default;

    virtual void write_data(const std::string& filename, const size_t i) = 0;

    StructureWriter(StructureDB& db) : stdb(db) {};

    virtual void set_precision(const size_t _p) { p = _p; w = p+6; };

  protected:
    StructureDB& stdb;
    double p = 10;  // output precision
    double w = p+6;  // column width
};
} // namespace mlip
} // namespace tadah
#endif // STRUCTURE_WRITER_H
