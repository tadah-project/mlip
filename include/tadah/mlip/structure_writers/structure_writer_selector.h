#ifndef STRUCTURE_WRITER_SELECTOR_H
#define STRUCTURE_WRITER_SELECTOR_H

#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/structure_writer.h>
#include <string>
#include <memory>
namespace tadah {
namespace mlip {

class StructureWriterSelector {
  public:
    /** The writer of @p type (CASTEP, LAMMPS or VASP, upper case) for @p db.
     *  @throws std::invalid_argument for any other type, naming it and the
     *  formats written. */
    static std::unique_ptr<StructureWriter> get_writer(const std::string& type, StructureDB& db);

};

} // namespace mlip
} // namespace tadah
#endif // STRUCTURE_WRITER_SELECTOR_H
