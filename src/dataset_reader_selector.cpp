#include <tadah/mlip/dataset_readers/dataset_reader_selector.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>
#include <tadah/mlip/dataset_readers/castep_geom_reader.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>

#include <fstream>
#include <iostream>

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
    std::cerr << "Unknown type! Returning nullptr." << std::endl;
    return nullptr;
  }

}

// Function to determine the file type based on content
std::string DatasetReaderSelector::determine_file_type_by_content(const std::string& filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << filepath << std::endl;
    return "Unknown file type";
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
