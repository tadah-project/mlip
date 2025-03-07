#include <tadah/mlip/structure_readers/castep_cell_reader.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>
#include <tadah/mlip/structure_readers/materials_project_reader.h>
#include <tadah/mlip/structure_readers/cod_reader.h>
#include <tadah/mlip/structure_readers/aflow_reader.h>
#include <tadah/mlip/key_storage/key_manager.h>
#include <tadah/mlip/structure_readers/nomad_reader.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

std::unique_ptr<IStructureReader>
StructureReaderSelector::getReader(const std::string &pathOrId) {
  std::string format = guessFormat(pathOrId);

  if (format == "MP") {
    std::string mpKey =
        KeyManager::getServiceKey(ServiceType::MP, /*allowPrompt=*/true);
    if (mpKey.empty()) {
      throw std::runtime_error(
          "No valid Materials Project key provided. Cannot continue.");
    }
    return std::make_unique<MaterialsProjectReader>(mpKey);
  } else if (format == "CIF") {
    return std::make_unique<CifReader>();
  } else if (format == "POSCAR") {
    return std::make_unique<VaspPoscarReader>();
  } else if (format == "CELL") {
    return std::make_unique<CastepCellReader>();
  }
  else if (format == "COD") {
      return std::make_unique<CODReader>();
  }
  else if (format == "AFLOW") {
      return std::make_unique<AflowReader>();
  }
  else if (format == "NOMAD") {
    return std::make_unique<NomadReader>();
  } else {
    throw std::runtime_error(
        "StructureReaderSelector::getReader - Unknown format for " + pathOrId);
  }
}

bool fileExists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}
bool isSevenDigits(const std::string &s) {
  if (s.size() != 7)
    return false;
  return std::all_of(s.begin(), s.end(), ::isdigit);
}
// A minimal heuristic for “NOMAD-like” ID
bool looksLikeNomadId(const std::string &s) {
  // Example: length typically ~27, so pick a range
  if (s.size() < 20 || s.size() > 40)
    return false;

  // Must be URL-safe base64-ish: A–Z, a–z, 0–9, -, _
  return std::all_of(s.begin(), s.end(), [](unsigned char c) {
    return (std::isalnum(c) != 0) || c == '-' || c == '_';
  });
}
std::string StructureReaderSelector::guessFormat(const std::string &pathOrId) {
  // 1) Check if path is a local file
  if (fileExists(pathOrId)) {
    // Heuristics by extension
    if (pathOrId.rfind("POSCAR", 0) == 0 || pathOrId.rfind("CONTCAR", 0) == 0) {
      return "POSCAR";
    }
    auto dotPos = pathOrId.find_last_of('.');
    if (dotPos != std::string::npos) {
      std::string ext = pathOrId.substr(dotPos + 1);
      std::transform(ext.begin(), ext.end(), ext.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (ext == "cif")
        return "CIF";
      if (ext == "cell")
        return "CELL";
    }
    // If no extension or no known extension
    return guessFormatFromContent(pathOrId);
  }

  // 2) Known online prefixes or patterns
  // MP uses mp- prefix: “mp-####”
  if (pathOrId.rfind("mp-", 0) == 0) {
    return "MP";
  }

  // COD uses 7-digit numeric code, e.g. "8104301".
  if (isSevenDigits(pathOrId)) {
    return "COD";
  }

  // AFLOW uses "aflow:######" style.
  if (pathOrId.rfind("aflow:", 0) == 0) {
    return "AFLOW";
  }

  // NOMAD minimal check
  if (looksLikeNomadId(pathOrId)) {
    return "NOMAD";
  }

  // 3) Fallback: guess-from-server approach

  // 4) Fail
  return "UNKNOWN";
}


bool isLikelyPoscar(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return false; // can't open -> can't detect
  }

  // Minimal number of lines for a typical POSCAR
  // 1) Title/Comment
  // 2) Scale factor
  // 3) Lattice vector line #1
  // 4) Lattice vector line #2
  // 5) Lattice vector line #3
  // 6) Element symbols or counts
  // 7) Possibly more lines (atom counts / Selective Dynamics / coordinates, etc.)

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(file, line)) {
    lines.push_back(line);
    if (lines.size() > 10) {
      // Reading more lines is okay, but we only need ~7 lines for detection
      break;
    }
  }
  file.close();

  // Must have ~6 lines to check
  if (lines.size() < 6) {
    return false;
  }

  // 1) line[0]: Title/comment (no strong requirement, can be anything)
  // 2) line[1]: Should parse as scale factor (double)
  {
    std::istringstream iss(lines[1]);
    double scale = 0.0;
    if (!(iss >> scale)) {
      // Could not parse a double
      return false;
    }
  }

  // 3) The next three lines (2..4) each should have 3 floating values
  for (int i = 2; i < 5; ++i) {
    std::istringstream iss(lines[i]);
    double x, y, z;
    if (!(iss >> x >> y >> z)) {
      // Could not parse 3 floats
      return false;
    }
  }

  // 4) line[5]: Often either element symbols or number of atoms
  //    But this line is optional. Let's do a minimal check:
  //    In many POSCARs, if it's not numeric, it might be e.g., "Si O" or "Fe" or something.
  //    We'll not strictly require anything here. Additional checks can be added if needed.

  // 5) Optional: Check for “Selective dynamics” or the “Direct/Cartesian” line.
  //    The line after the element/atom count might be "Selective dynamics"
  //    or "s" in some usage. Then the next line might be "Direct" or "Cartesian".
  
  // If we've gotten this far, the file strongly resembles a POSCAR.
  return true;
}

// A minimal function to see if “text” starts with “prefix” (case-sensitive).
static bool startsWith(const std::string &text, const std::string &prefix) {
  return text.rfind(prefix, 0) == 0;
}

bool isLikelyCif(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return false; // Could not open -> not a CIF
  }

  // We'll read up to ~30 lines (CIF can have many lines, but this is enough to detect).
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
    if (lines.size() >= 30) {
      break;
    }
  }
  file.close();

  // Basic heuristic checks:
  // 1) Is there at least one line starting with "data_"?
  // 2) Are there lines with typical CIF tags, like "_cell_length_", "_cell_angle_", "_atom_site_", etc.?

  bool foundDataLine = false;
  bool foundCellOrAtomLine = false;

  for (auto &ln : lines) {
    // Trim leading/trailing whitespace for easier checks
    // (could also keep them if your checking is simple)
    auto startPos = ln.find_first_not_of(" \t\r\n");
    if (startPos == std::string::npos) {
      continue; // empty line
    }
    std::string trimmed = ln.substr(startPos);

    if (startsWith(trimmed, "data_")) {
      foundDataLine = true;
    }
    if (startsWith(trimmed, "_cell_length_") ||
        startsWith(trimmed, "_cell_angle_")  ||
        startsWith(trimmed, "_symmetry_")    ||
        startsWith(trimmed, "_atom_site_"))
    {
      foundCellOrAtomLine = true;
    }
  }

  // A very minimal criterion: must have a data_ line plus at least one known CIF tag line.
  if (foundDataLine && foundCellOrAtomLine) {
    return true;
  }

  // Possibly check for “loop_” lines if you want extra confidence.
  return false;
}

// Utility: checks if a line begins with a (case-insensitive) prefix
static bool startsWithIgnoreCase(const std::string& line, const std::string& prefix) {
  if (line.size() < prefix.size()) return false;
  
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(line[i])) !=
        std::tolower(static_cast<unsigned char>(prefix[i]))) {
      return false;
    }
  }
  return true;
}

bool isLikelyCell(const std::string &filepath) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    return false;
  }

  // Read up to, say, 50 lines. CASTEP .cell files can be quite large,
  // but typical detection should be near the top.
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
    if (lines.size() >= 50) {
      break;
    }
  }
  file.close();

  // We look for e.g. %BLOCK LATTICE_CART, %BLOCK LATTICE_ABC, or %BLOCK POSITIONS_FRAC.
  bool foundLatticeBlock = false;
  bool foundPositionsBlock = false;

  for (const auto &ln : lines) {
    // Trim leading/trailing whitespace
    auto startPos = ln.find_first_not_of(" \t");
    if (startPos == std::string::npos) {
      continue; // empty line
    }
    std::string trimmed = ln.substr(startPos);

    // Check for “%BLOCK LATTICE_CART” or “%BLOCK LATTICE_ABC” ignoring case
    if (startsWithIgnoreCase(trimmed, "%BLOCK LATTICE_CART") ||
        startsWithIgnoreCase(trimmed, "%BLOCK LATTICE_ABC"))
    {
      foundLatticeBlock = true;
    }
    // Also check if it contains e.g. “%BLOCK POSITIONS_FRAC”
    if (startsWithIgnoreCase(trimmed, "%BLOCK POSITIONS_FRAC") ||
        startsWithIgnoreCase(trimmed, "%BLOCK POSITIONS_ABS"))
    {
      foundPositionsBlock = true;
    }
  }

  // A minimal “likely .cell” detection: we found a lattice block and a positions block
  // Very small files might omit one or the other, but this is typically robust.
  if (foundLatticeBlock && foundPositionsBlock) {
    return true;
  }

  // If not found, it might still be a .cell file, but the heuristics do not confirm it.
  return false;
}

std::string
StructureReaderSelector::guessFormatFromContent(const std::string &filePath) {
  std::ifstream ifs(filePath);

  std::string line;
  if (std::getline(ifs, line)) {
    if (isLikelyCif(filePath)) {
      return "CIF";
    }
    if (isLikelyCell(filePath)) {
      return "CELL";
    }
    if (isLikelyPoscar(filePath)) {
      return "POSCAR";
    }
  }

  return "UNKNOWN";
}
