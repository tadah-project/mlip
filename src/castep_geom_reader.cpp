#include <tadah/core/utils.h>
#include <tadah/mlip/dataset_readers/castep_geom_reader.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>
namespace tadah {
namespace mlip {

CastepGeomReader::CastepGeomReader(StructureDB& db)
: CastepMDReader(db) {}

CastepGeomReader::CastepGeomReader(StructureDB& db, const std::string& filename)
: CastepMDReader(db, filename) {}

std::string CastepGeomReader::get_first_label(std::string &step) {
  return "Filename: "+filename_+ " | Units: (eV, Angstrom) | Step: " + step;
}
std::string CastepGeomReader::get_label(std::string &step) {
  return "Step: " + step;
}
} // namespace mlip
} // namespace tadah
