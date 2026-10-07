#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tadah/core/number_format.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>
#include <vector>
namespace tadah {
namespace mlip {

// A small helper to parse each line into tokens
static std::vector<std::string> split(const std::string &str);

// The digits of a POSCAR atom count: the token after at most one '+' (as
// VASP's integer read, ASE and pymatgen accept "+2"), or an empty view when
// the rest is not digits only (a '-', a fraction, a word).
static std::string_view count_digits(const std::string &token) {
  std::string_view digits(token);
  if (!digits.empty() && digits.front() == '+') digits.remove_prefix(1);
  if (digits.empty() ||
      !std::all_of(digits.begin(), digits.end(),
                   [](unsigned char c) { return std::isdigit(c) != 0; }))
    return {};
  return digits;
}

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
  // Every error names the file. The structure is built in a local and
  // replaces the previous one only when the file has been read.
  Structure st;
  try {
    if (lines.size() < 8) {
      throw std::runtime_error("not enough lines for a POSCAR file (" +
                               std::to_string(lines.size()) +
                               "; it has 8 at least).");
    }
    st = parsePoscar(lines);
  } catch (const std::exception &e) {
    throw std::runtime_error("VaspPoscarReader: POSCAR file " + path + ": " +
                             e.what());
  }
  st.label = "POSCAR file: " + path;
  m_structure = std::move(st);
}

Structure VaspPoscarReader::getStructure() const { return m_structure; }

Structure VaspPoscarReader::parsePoscar(const std::vector<std::string> &lines) {
  Structure st;
  // 1) Ignore comment line (lines[0])
  // 2) Scaling factor(s), VASP semantics: one number s > 0 scales the
  //    lattice (and Cartesian coordinates); s < 0 is the target cell volume
  //    (|s| = det of the scaled lattice); three numbers s_x s_y s_z > 0
  //    scale the x, y and z components of the lattice vectors (and of
  //    Cartesian coordinates). Anything else is an error.
  std::array<double, 3> scale{1.0, 1.0, 1.0};
  bool volumeScale = false;
  {
    std::vector<double> nums;
    for (const auto &tok : split(lines[1])) {
      char *end = nullptr;
      const double v = std::strtod(tok.c_str(), &end);
      if (end == tok.c_str() || *end != '\0') break;  // comment follows
      nums.push_back(v);
    }
    if (nums.size() == 1) {
      if (nums[0] == 0.0 || !std::isfinite(nums[0])) {
        throw std::runtime_error(
          "the scaling factor on line 2 is " +
          lines[1] + "; it must be non-zero (a negative value is the cell "
          "volume).");
      }
      volumeScale = nums[0] < 0.0;
      scale = {nums[0], nums[0], nums[0]};
    } else if (nums.size() == 3) {
      for (double v : nums) {
        if (!(v > 0.0) || !std::isfinite(v)) {
          throw std::runtime_error(
            "the three scaling factors on line 2 ('" +
            lines[1] + "') must all be positive.");
        }
      }
      scale = {nums[0], nums[1], nums[2]};
    } else {
      throw std::runtime_error(
        "line 2 must hold one scaling factor or three "
        "(one per Cartesian direction), got '" + lines[1] + "'.");
    }
  }

  // 3-5) Lattice vectors: line 3+i is lattice vector a_i, stored as row i.
  for (int i = 0; i < 3; ++i) {
    const auto f = split(lines[2 + i]);
    if (f.size() < 3) {
      throw std::runtime_error(
        "lattice vector line " + std::to_string(3 + i) +
        " needs three numbers.");
    }
    for (int j = 0; j < 3; ++j)
      st.cell(i, j) = tadah::core::to_double(f[j]);
  }
  if (volumeScale) {
    const double det = std::fabs(st.get_volume());
    if (det == 0.0) {
      throw std::runtime_error(
        "the lattice vectors are coplanar; a negative "
        "scaling factor (a target volume) cannot be applied.");
    }
    const double s = std::cbrt(-scale[0] / det);
    scale = {s, s, s};
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      st.cell(i, j) *= scale[j];

  // 6) Next line could be element symbols or counts
  // For simplicity, assume it's element symbols
  bool hasElementSymbols = true;
  std::vector<std::string> fields = split(lines[5]);
  // If they're all atom counts -> that's the counts line, no symbols
  {
    bool allCounts = true;
    for (auto &f : fields) {
      if (count_digits(f).empty()) {
        allCounts = false;
        break;
      }
    }
    if (allCounts)
      hasElementSymbols = false;
  }

  // If hasElementSymbols is true, the line after it is counts.
  // If not, line[5] is counts
  size_t countsLineIndex = hasElementSymbols ? 6 : 5;
  if (countsLineIndex >= lines.size()) {
    throw std::runtime_error(
      "incomplete POSCAR (missing counts line).");
  }

  std::vector<int> speciesCounts;
  {
    std::vector<std::string> cfields = split(lines[countsLineIndex]);
    for (auto &cf : cfields) {
      const std::string where = "the atom count '" + cf + "' on line " +
                                std::to_string(countsLineIndex + 1);
      const std::string_view digits = count_digits(cf);
      if (digits.empty())
        throw std::runtime_error(where + " is not a whole number.");
      int count = 0;
      const auto r =
          std::from_chars(digits.data(), digits.data() + digits.size(), count);
      if (r.ec == std::errc::result_out_of_range)
        throw std::runtime_error(where + " is too large.");
      if (r.ec != std::errc{} || r.ptr != digits.data() + digits.size())
        throw std::runtime_error(where + " is not a whole number.");
      speciesCounts.push_back(count);
    }
  }
  size_t nextLineIndex = countsLineIndex + 1;
  if (nextLineIndex >= lines.size()) {
    throw std::runtime_error(
      "incomplete POSCAR (no coords section).");
  }

  // VASP decides from the first character of a line: an optional line
  // starting with S/s is "Selective dynamics"; then C, c, K or k selects
  // Cartesian coordinates and anything else Direct (fractional) ones.
  auto first_char = [](const std::string &line) {
    const auto p = line.find_first_not_of(" \t\r");
    return p == std::string::npos
               ? '\0'
               : static_cast<char>(std::tolower(
                     static_cast<unsigned char>(line[p])));
  };
  if (first_char(lines[nextLineIndex]) == 's') {
    nextLineIndex++;
  }
  bool directCoords = true;
  {
    if (nextLineIndex >= lines.size()) {
      throw std::runtime_error("no coordinate type line.");
    }
    const char c = first_char(lines[nextLineIndex]);
    if (c == 'c' || c == 'k') {
      directCoords = false;
    }
    nextLineIndex++;
  }

  // Now we read atomic coordinates
  // sum(speciesCounts) is total # of atoms
  int totalAtoms = 0;
  for (auto c : speciesCounts)
  totalAtoms += c;

  if (nextLineIndex + totalAtoms > lines.size()) {
    throw std::runtime_error(
      "not enough lines for all atoms.");
  }

  // If hasElementSymbols, we have e.g. line[5] = "Si O", line[6] = "2 4"
  // We might store them in an array to track species assignment
  std::vector<std::string> elemSymbols;
  if (hasElementSymbols) {
    elemSymbols = split(lines[5]);
    // VASP 6 writes POTCAR names here ('Ti_pv/8c1b2d3e', 'Si/b0d4a2'): the
    // element is the part before '/' and '_'.
    for (auto &token : elemSymbols) {
      std::string element = token.substr(0, token.find('/'));
      element = element.substr(0, element.find('_'));
      try {
        tadah::core::PeriodicTable::find_by_symbol(element);
      } catch (const std::exception &) {
        throw std::runtime_error(
          "'" + token + "' on line 6 is not an element "
          "symbol (or a VASP POTCAR name such as Ti_pv or Si/b0d4a2).");
      }
      token = element;
    }
  } else {
    throw std::runtime_error(
      "line 6 holds the atom counts, not the element "
      "symbols (the VASP 4 layout); add the element-symbol line above the "
      "counts.");
  }
  if (elemSymbols.size() != speciesCounts.size()) {
    throw std::runtime_error(
      std::to_string(elemSymbols.size()) +
      " element symbols on line 6 but " + std::to_string(speciesCounts.size()) +
      " atom counts on line 7.");
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
        "not enough coords for atom line " +
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
      double cx = x * st.cell(0, 0) + y * st.cell(1, 0) +
        z * st.cell(2, 0);
      double cy = x * st.cell(0, 1) + y * st.cell(1, 1) +
        z * st.cell(2, 1);
      double cz = x * st.cell(0, 2) + y * st.cell(1, 2) +
        z * st.cell(2, 2);
      a.position[0] = cx;
      a.position[1] = cy;
      a.position[2] = cz;
    } else {
      // Cartesian coordinates carry the same scaling as the lattice.
      a.position[0] = x * scale[0];
      a.position[1] = y * scale[1];
      a.position[2] = z * scale[2];
    }

    st.atoms.push_back(a);
  }
  return st;
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
