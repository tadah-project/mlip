#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>
#include <tadah/mlip/structure_readers/castep_cell_reader.h>
/*#include <tadah/mlip/structure_readers/extended_xyz_reader.h>*/
/*#include <tadah/mlip/structure_readers/xsf_reader.h>*/
#include <tadah/mlip/structure_readers/materials_project_reader.h>
/*#include <tadah/mlip/structure_readers/cod_reader.h>*/
/*#include <tadah/mlip/structure_readers/aflow_reader.h>*/
/*#include <tadah/mlip/structure_readers/nomad_reader.h>*/

#include <fstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

std::unique_ptr<StructureReader> StructureReaderSelector::getReader(const std::string& pathOrId)
{
  std::cout << "Reading structure from: " << pathOrId << std::endl;
    std::string format = guessFormat(pathOrId);
    std::cout << "Format: " << format << std::endl;

    if (format == "MP") {
        return std::make_unique<MaterialsProjectReader>("MY_MP_API_KEY"); // Provide your real key
    }
    else if (format == "CIF") {
      std::cout << "CIF Reader Selected" << std::endl;
        return std::make_unique<CifReader>();
    }
    else if (format == "POSCAR") {
        return std::make_unique<VaspPoscarReader>();
    }
    else if (format == "CELL") {
        return std::make_unique<CastepCellReader>();
    }
    /*else if (format == "XYZ") {*/
    /*    return std::make_unique<ExtendedXyzReader>();*/
    /*}*/
    /*else if (format == "XSF") {*/
    /*    return std::make_unique<XsfReader>();*/
    /*}*/
    /*else if (format == "COD") {*/
    /*    return std::make_unique<CODReader>();*/
    /*}*/
    /*else if (format == "AFLOW") {*/
    /*    return std::make_unique<AflowReader>();*/
    /*}*/
    /*else if (format == "NOMAD") {*/
    /*    return std::make_unique<NomadReader>();*/
    /*}*/
    else {
        throw std::runtime_error("StructureReaderSelector::getReader - Unknown format for " + pathOrId);
    }
}

std::string StructureReaderSelector::guessFormat(const std::string& pathOrId)
{
    // Check known online ID patterns:
    if (pathOrId.rfind("mp-", 0) == 0) {
        return "MP";
    }
    if (pathOrId.rfind("cod-", 0) == 0) {
        return "COD";
    }
    if (pathOrId.rfind("aflow-", 0) == 0) {
        return "AFLOW";
    }
    if (pathOrId.rfind("nomad-", 0) == 0) {
        return "NOMAD";
    }

    // Check file extension:
    auto dotPos = pathOrId.find_last_of('.');
    if (dotPos != std::string::npos) {
        std::string ext = pathOrId.substr(dotPos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (ext == "cif")  return "CIF";
        if (ext == "cell") return "CELL";
        if (ext == "xsf")  return "XSF";
        if (ext == "xyz")  return "XYZ";
        // Some VASP files may not have an extension
    }

    return guessFormatFromContent(pathOrId);
}

std::string StructureReaderSelector::guessFormatFromContent(const std::string& filePath)
{
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        // If can't open, guess POSCAR (commonly no extension).
        return "POSCAR";
    }

    std::string line;
    if (std::getline(ifs, line)) {
        // Some naive checks
        if (line.find("lattice_vector") != std::string::npos) {
            return "XSF";
        }
        if (line.find("CELL_PARAMETERS") != std::string::npos) {
            return "CIF";
        }
        if (line.find("%BLOCK LATTICE_") != std::string::npos) {
            return "CELL";
        }
    }

    // fallback
    return "POSCAR";
}

