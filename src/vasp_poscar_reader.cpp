#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <tadah/core/number_format.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>
#include <vector>
namespace tadah {
namespace mlip {

// A small helper to parse each line into tokens
static std::vector<std::string> split(const std::string &str);

void VaspPoscarReader::read(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("VaspPoscarReader::read: Cannot open " + path);
  }

  std::vector<std::string> lines;
  {
    std::string line;
    while (std::getline(ifs, line)) {
      lines.push_back(line);
    }
  }
  if (lines.size() < 8) {
    throw std::runtime_error(
      "VaspPoscarReader::read: Not enough lines for a POSCAR file.");
  }

  parsePoscar(lines);
  m_structure.label = "POSCAR file: " + path;
}

Structure VaspPoscarReader::getStructure() const { return m_structure; }

void VaspPoscarReader::parsePoscar(const std::vector<std::string> &lines) {
  // 1) Ignore comment line (lines[0])
  // 2) Scaling factor
  double scaleFactor = 1.0;
  {
    std::stringstream ss(lines[1]);
    ss >> scaleFactor;
    if (!ss.good() && !ss.eof()) {
      // if parse fails, use default = 1.0
      scaleFactor = 1.0;
    }
  }

  // 3-5) Lattice vectors
  for (int i = 0; i < 3; ++i) {
    std::stringstream ss(lines[2 + i]);
    double x, y, z;
    ss >> x >> y >> z;
    m_structure.cell(i, 0) = x * scaleFactor;
    m_structure.cell(i, 1) = y * scaleFactor;
    m_structure.cell(i, 2) = z * scaleFactor;
  }

  // 6) Next line could be element symbols or counts
  // For simplicity, assume it's element symbols
  bool hasElementSymbols = true;
  std::vector<std::string> fields = split(lines[5]);
  // If they're all digits -> that's the counts line, no symbols
  {
    bool allDigits = true;
    for (auto &f : fields) {
      if (f.empty() || !std::all_of(f.begin(), f.end(), ::isdigit)) {
        allDigits = false;
        break;
      }
    }
    if (allDigits)
      hasElementSymbols = false;
  }

  // If hasElementSymbols is true, the line after it is counts.
  // If not, line[5] is counts
  size_t countsLineIndex = hasElementSymbols ? 6 : 5;
  if (countsLineIndex >= lines.size()) {
    throw std::runtime_error(
      "VaspPoscarReader: incomplete POSCAR (missing counts line).");
  }

  std::vector<int> speciesCounts;
  {
    std::vector<std::string> cfields = split(lines[countsLineIndex]);
    for (auto &cf : cfields) {
      speciesCounts.push_back(std::stoi(cf));
    }
  }
  size_t nextLineIndex = countsLineIndex + 1;
  if (nextLineIndex >= lines.size()) {
    throw std::runtime_error(
      "VaspPoscarReader: incomplete POSCAR (no coords section).");
  }

  // Check if there's a "Selective Dynamics" line
  // bool selectiveDynamics = false;
  bool directCoords = true; // or cart
  {
    std::string s = lines[nextLineIndex];
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s.find("selective") != std::string::npos) {
      // selectiveDynamics = true;
      nextLineIndex++;
    }
  }
  // Now check direct or cart
  {
    if (nextLineIndex >= lines.size()) {
      throw std::runtime_error("VaspPoscarReader: no coordinate type line.");
    }
    std::string s = lines[nextLineIndex];
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s.find("cart") != std::string::npos) {
      directCoords = false;
    }
    // else direct if "direct" or "frac"
    nextLineIndex++;
  }

  // Now we read atomic coordinates
  // sum(speciesCounts) is total # of atoms
  int totalAtoms = 0;
  for (auto c : speciesCounts)
  totalAtoms += c;

  if (nextLineIndex + totalAtoms > lines.size()) {
    throw std::runtime_error(
      "VaspPoscarReader: Not enough lines for all atoms.");
  }

  // If hasElementSymbols, we have e.g. line[5] = "Si O", line[6] = "2 4"
  // We might store them in an array to track species assignment
  std::vector<std::string> elemSymbols;
  if (hasElementSymbols) {
    elemSymbols = split(lines[5]);
  } else {
    // fallback: unknown symbol, use X
    int nEl = (int)speciesCounts.size();
    elemSymbols.resize(nEl, "X");
  }

  // Expand element symbols for each atomic site
  std::vector<std::string> atomSymbols;
  {
    size_t idx = 0;
    for (size_t iEl = 0; iEl < speciesCounts.size(); iEl++) {
      for (int k = 0; k < speciesCounts[iEl]; k++) {
        atomSymbols.push_back(elemSymbols[iEl]);
        idx++;
      }
    }
  }

  // parse coords
  for (int i = 0; i < totalAtoms; i++) {
    std::vector<std::string> cfields = split(lines[nextLineIndex + i]);
    if (cfields.size() < 3) {
      throw std::runtime_error(
        "VaspPoscarReader: not enough coords for atom line " +
        std::to_string(nextLineIndex + i));
    }
    // core::to_double: std::stod, but an underflowing literal (a subnormal
    // coordinate, as the POSCAR export writes it) is a value, not an error.
    double x = tadah::core::to_double(cfields[0]);
    double y = tadah::core::to_double(cfields[1]);
    double z = tadah::core::to_double(cfields[2]);

    Atom a;
    // parse element from atomSymbols[i] if you maintain a map
    // For brevity, store minimal approach:
    tadah::core::Element e = tadah::core::PeriodicTable().find_by_symbol(atomSymbols[i]);
    static_cast<tadah::core::Element &>(a) = e;

    if (directCoords) {
      // Convert fractional -> cart
      double cx = x * m_structure.cell(0, 0) + y * m_structure.cell(1, 0) +
        z * m_structure.cell(2, 0);
      double cy = x * m_structure.cell(0, 1) + y * m_structure.cell(1, 1) +
        z * m_structure.cell(2, 1);
      double cz = x * m_structure.cell(0, 2) + y * m_structure.cell(1, 2) +
        z * m_structure.cell(2, 2);
      a.position[0] = cx;
      a.position[1] = cy;
      a.position[2] = cz;
    } else {
      // Already cart
      a.position[0] = x;
      a.position[1] = y;
      a.position[2] = z;
    }

    m_structure.atoms.push_back(a);
  }
}

static std::vector<std::string> split(const std::string &str) {
  std::vector<std::string> tokens;
  std::istringstream iss(str);
  std::string s;
  while (iss >> s) {
    tokens.push_back(s);
  }
  return tokens;
}
} // namespace mlip
} // namespace tadah
