#include <tadah/core/utils.h>
#include <tadah/mlip/dataset_readers/castep_geom_reader.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>

CastepGeomReader::CastepGeomReader(StructureDB& db)
: CastepMDReader(db) {}

CastepGeomReader::CastepGeomReader(StructureDB& db, const std::string& filename)
: CastepMDReader(db, filename) {}

double CastepGeomReader::calc_P_ideal(Structure &, double ) {
  return 0; // dummy as geom stress tensor os virial
}

std::string CastepGeomReader::get_first_label(std::string &step) {
  return "Filename: "+filename_+ " | Units: (eV, Angstrom) | Step: " + step;
}
std::string CastepGeomReader::get_label(std::string &step) {
  return "Step: " + step;
}
