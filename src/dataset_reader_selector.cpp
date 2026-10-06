#include <tadah/mlip/dataset_readers/dataset_reader_selector.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>
#include <tadah/mlip/dataset_readers/castep_geom_reader.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
namespace tadah {
namespace mlip {

namespace {
const char *const kRecognisedDftOutputs =
    "VASP OUTCAR, vasprun.xml, CASTEP .castep, .md, .geom";
}

// Factory method implementation
std::unique_ptr<DatasetReader> DatasetReaderSelector::get_reader(const std::string& filepath, StructureDB& db) {

  std::string type = determine_file_type_by_content(filepath);

  if (type == "CASTEP.CASTEP") {
    return std::make_unique<CastepCastepReader>(db,filepath);
  } else if (type == "CASTEP.MD") {
    return std::make_unique<CastepMDReader>(db,filepath);
  } else if (type == "CASTEP.GEOM") {
    return std::make_unique<CastepGeomReader>(db,filepath);
  } else if (type == "VASP.VASPRUN") {
    return std::make_unique<VaspVasprunReader>(db,filepath);
  } else if (type == "VASP.OUTCAR") {
    return std::make_unique<VaspOutcarReader>(db,filepath);
  } else {
    throw std::runtime_error(
        "'" + filepath + "' is not a recognised DFT output (" +
        kRecognisedDftOutputs + "); structure files such as POSCAR, CIF or "
        ".cell are read with -s by tadah data print, data write and predict");
  }

}

// Function to determine the file type based on content
std::string DatasetReaderSelector::determine_file_type_by_content(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("cannot open '" + filepath +
                             "' (recognised DFT outputs: " +
                             kRecognisedDftOutputs + ")");
  }

  std::string line;
  while (std::getline(file, line)) {

    if (line.find("incar:") != std::string::npos ||  line.find("OUTCAR:") != std::string::npos ||
      line.find("outcar") != std::string::npos || line.find("POTCAR:") != std::string::npos) {
      return "VASP.OUTCAR";
    } else if (line.find("<modeling>") != std::string::npos || line.find("<calculation>") != std::string::npos) {
      return "VASP.VASPRUN";
    }
    else if (line.find("<-- c") != std::string::npos) {
      return "CASTEP.GEOM";
    } else if (line.find("<-- E") != std::string::npos) {
      // <-- E is also in .geom but .geom has <-- c before <-- E
      return "CASTEP.MD";
    } else if (line.find("Unit Cell") != std::string::npos) { 
      return "CASTEP.CASTEP";
    }
  }

  return "Unknown file type";
}
} // namespace mlip
} // namespace tadah
