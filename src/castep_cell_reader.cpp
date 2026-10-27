#include <tadah/mlip/dataset_readers/castep_species.h>
#include <tadah/mlip/structure_readers/castep_cell_reader.h>
#include <tadah/core/number_format.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace tadah {
namespace mlip {

//--------------------------------------------------------------------
// Helper to split a line by whitespace
//--------------------------------------------------------------------
static std::vector<std::string> split_line(const std::string &line) {
  std::istringstream iss(line);
  std::vector<std::string> tokens;
  std::string s;
  while (iss >> s) {
    tokens.push_back(s);
  }
  return tokens;
}

// Convert degrees to radians
static double deg2rad(double deg) {
  static const double PI = 3.141592653589793;
  return deg * (PI / 180.0);
}

//--------------------------------------------------------------------
// Implementation of public methods
//--------------------------------------------------------------------
void CastepCellReader::read(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("the structure file " + path + " cannot be opened");
  }

  std::ostringstream buffer;
  buffer << ifs.rdbuf();
  // Every error names the file. The structure is built in a local and
  // replaces the previous one only when the file has been read.
  Structure st;
  try {
    st = parseCellContents(buffer.str());
  } catch (const std::exception &e) {
    throw std::runtime_error("CASTEP cell file " + path +
                             ": " + e.what());
  }
  st.label = "CASTEP cell file: " + path;
  m_structure = std::move(st);
}

Structure CastepCellReader::getStructure() const {
  return m_structure;
}

//--------------------------------------------------------------------
// lengthUnit: the factor that converts a CASTEP length unit to Angstrom.
// CASTEP accepts an optional unit line as the first line of LATTICE_CART,
// LATTICE_ABC and POSITIONS_ABS (default: ang).
//--------------------------------------------------------------------
static double lengthUnit(const std::string &word, const std::string &block) {
  std::string u = word;
  std::transform(u.begin(), u.end(), u.begin(), ::tolower);
  if (u == "ang") return 1.0;
  if (u == "bohr" || u == "a0") return 0.529177210903;
  if (u == "nm") return 10.0;
  if (u == "cm") return 1e8;
  if (u == "m") return 1e10;
  if (u == "pm") return 1e-2;
  throw std::runtime_error("unknown length unit '" + word +
                           "' in " + block +
                           " (known: ang, bohr, a0, nm, cm, m, pm).");
}

// CASTEP ignores everything after '!', '#' or ';' on a line (species
// labels and numbers never contain them).
static std::string strip_comment(const std::string &line) {
  return line.substr(0, line.find_first_of("!#;"));
}

// True for a line with nothing but blanks.
static bool blank(const std::string &line) {
  return line.find_first_not_of(" \t\r") == std::string::npos;
}

//--------------------------------------------------------------------
// parseCellContents
//
// The blocks may come in any order: their lines are collected first, and
// fractional positions are converted only once the lattice is known.
//--------------------------------------------------------------------
Structure CastepCellReader::parseCellContents(const std::string &contents) {
  Structure st;
  std::istringstream iss(contents);
  std::string line;

  enum class Block { None, LatticeCart, LatticeABC, PosFrac, PosAbs };
  Block block = Block::None;
  bool seenCart = false, seenABC = false, seenFrac = false, seenAbs = false;
  std::vector<std::string> cartLines, abcLines, fracLines, absLines;

  auto block_name = [](const std::string &lcase, const std::string &kw) {
    // "%block <name>" / "%endblock <name>", any spacing
    std::istringstream ls(lcase);
    std::string k, name;
    ls >> k >> name;
    return k == kw ? name : std::string();
  };

  while (std::getline(iss, line)) {
    line = strip_comment(line);
    std::string lcase = line;
    std::transform(lcase.begin(), lcase.end(), lcase.begin(), ::tolower);

    const std::string opened = block_name(lcase, "%block");
    if (!opened.empty()) {
      if (opened == "lattice_cart") { block = Block::LatticeCart; seenCart = true; }
      else if (opened == "lattice_abc") { block = Block::LatticeABC; seenABC = true; }
      else if (opened == "positions_frac") { block = Block::PosFrac; seenFrac = true; }
      else if (opened == "positions_abs") { block = Block::PosAbs; seenAbs = true; }
      else block = Block::None;  // a block this reader does not use
      continue;
    }
    if (!block_name(lcase, "%endblock").empty()) {
      block = Block::None;
      continue;
    }
    if (blank(line)) continue;
    switch (block) {
      case Block::LatticeCart: cartLines.push_back(line); break;
      case Block::LatticeABC:  abcLines.push_back(line);  break;
      case Block::PosFrac:     fracLines.push_back(line); break;
      case Block::PosAbs:      absLines.push_back(line);  break;
      case Block::None:        break;
    }
  }

  // An optional first line with a single word is the block's length unit.
  auto take_unit = [](std::vector<std::string> &lines,
                      const std::string &name) {
    if (!lines.empty()) {
      const auto toks = split_line(lines.front());
      if (toks.size() == 1) {
        const double f = lengthUnit(toks[0], name);
        lines.erase(lines.begin());
        return f;
      }
    }
    return 1.0;
  };

  // ---- lattice ----
  if (seenCart && seenABC) {
    throw std::runtime_error(
      "both LATTICE_CART and LATTICE_ABC are given.");
  }
  if (!seenCart && !seenABC) {
    throw std::runtime_error(
      "no lattice block (LATTICE_CART or LATTICE_ABC).");
  }
  if (seenCart) {
    const double unit = take_unit(cartLines, "LATTICE_CART");
    if (cartLines.size() != 3) {
      throw std::runtime_error(
        "LATTICE_CART needs exactly three lattice "
        "vectors, found " + std::to_string(cartLines.size()) + ".");
    }
    for (int i = 0; i < 3; ++i) {
      const auto toks = split_line(cartLines[i]);
      if (toks.size() < 3) {
        throw std::runtime_error(
          "LATTICE_CART line missing coords: '" +
          cartLines[i] + "'.");
      }
      // Line i is lattice vector a_i, stored as row i.
      for (int j = 0; j < 3; ++j)
        st.cell(i, j) = unit * tadah::core::to_double(toks[j]);
    }
  } else {
    const double unit = take_unit(abcLines, "LATTICE_ABC");
    parseLatticeABC(abcLines, unit, st);
  }
  {
    const tadah::core::Vec3d a0 = st.cell.row(0);
    const tadah::core::Vec3d a1 = st.cell.row(1);
    const tadah::core::Vec3d a2 = st.cell.row(2);
    const double det = a0 * a1.cross(a2);
    const double scale = a0.norm() * a1.norm() * a2.norm();
    if (!(std::fabs(det) > 1e-12 * scale)) {
      throw std::runtime_error(
        "the lattice vectors are coplanar (singular "
        "cell).");
    }
  }

  // ---- positions ----
  if (seenFrac && seenAbs) {
    throw std::runtime_error(
      "both POSITIONS_FRAC and POSITIONS_ABS are given.");
  }
  if (!seenFrac && !seenAbs) {
    throw std::runtime_error(
      "no positions block (POSITIONS_FRAC or "
      "POSITIONS_ABS).");
  }
  auto atom_line = [&](const std::string &ln, const std::string &name) {
    const auto toks = split_line(ln);
    if (toks.size() < 4) {
      throw std::runtime_error(name + " line needs a species and three "
                               "coordinates: '" + ln + "'.");
    }
    return toks;
  };
  for (const auto &ln : fracLines) {
    const auto toks = atom_line(ln, "POSITIONS_FRAC");
    double cx, cy, cz;
    fracToAbs(st, tadah::core::to_double(toks[1]),
              tadah::core::to_double(toks[2]), tadah::core::to_double(toks[3]),
              cx, cy, cz);
    Atom a;
    static_cast<tadah::core::Element&>(a) = parseSpecies(toks[0]);
    a.position = {cx, cy, cz};
    st.atoms.push_back(a);
  }
  if (seenAbs) {
    const double unit = take_unit(absLines, "POSITIONS_ABS");
    for (const auto &ln : absLines) {
      const auto toks = atom_line(ln, "POSITIONS_ABS");
      Atom a;
      static_cast<tadah::core::Element&>(a) = parseSpecies(toks[0]);
      a.position = {unit * tadah::core::to_double(toks[1]),
                    unit * tadah::core::to_double(toks[2]),
                    unit * tadah::core::to_double(toks[3])};
      st.atoms.push_back(a);
    }
  }
  if (st.atoms.empty()) {
    throw std::runtime_error(
      std::string("the ") +
      (seenFrac ? "POSITIONS_FRAC" : "POSITIONS_ABS") +
      " block has no atoms.");
  }
  return st;
}

//--------------------------------------------------------------------
// parseLatticeABC
//--------------------------------------------------------------------
void CastepCellReader::parseLatticeABC(const std::vector<std::string> &lines,
                                       double unit, Structure &st) {
  // Two lines (the unit line already removed): "a b c" and
  // "alpha beta gamma" (degrees).
  if (lines.size() != 2 || split_line(lines[0]).size() < 3 ||
      split_line(lines[1]).size() < 3) {
    throw std::runtime_error(
      "LATTICE_ABC block incomplete (need a line a b c "
      "and a line alpha beta gamma).");
  }
  const auto l0 = split_line(lines[0]);
  const auto l1 = split_line(lines[1]);
  const double aVal = unit * tadah::core::to_double(l0[0]);
  const double bVal = unit * tadah::core::to_double(l0[1]);
  const double cVal = unit * tadah::core::to_double(l0[2]);
  const double alphaDeg = tadah::core::to_double(l1[0]);
  const double betaDeg  = tadah::core::to_double(l1[1]);
  const double gammaDeg = tadah::core::to_double(l1[2]);

  double alpha = deg2rad(alphaDeg);
  double beta  = deg2rad(betaDeg);
  double gamma = deg2rad(gammaDeg);

  // CASTEP's orientation (rows are the lattice vectors):
  //   a = (aVal, 0, 0)
  //   b = (bVal cos(gamma), bVal sin(gamma), 0)
  //   c = (cVal cos(beta), cVal [cos(alpha) - cos(beta) cos(gamma)] / sin(gamma),
  //        sqrt(cVal^2 - cx^2 - cy^2))

  // a
  st.cell(0,0) = aVal;
  st.cell(0,1) = 0.0;
  st.cell(0,2) = 0.0;

  // b
  double bx = bVal * std::cos(gamma);
  double by = bVal * std::sin(gamma);
  st.cell(1,0) = bx;
  st.cell(1,1) = by;
  st.cell(1,2) = 0.0;

  // c
  double cx = cVal * std::cos(beta);
  double cy = cVal * (std::cos(alpha) - std::cos(beta)*std::cos(gamma)) / std::sin(gamma);
  double czsq = cVal*cVal - cx*cx - cy*cy;
  if (czsq < 0.0) {
    // could be numerical rounding or truly invalid
    if (czsq > -1e-10) czsq = 0.0; // clamp small negative to zero
    else {
      throw std::runtime_error("invalid angles produce negative c.z^2.");
    }
  }
  double cz = std::sqrt(czsq);

  st.cell(2,0) = cx;
  st.cell(2,1) = cy;
  st.cell(2,2) = cz;
}

//--------------------------------------------------------------------
// fracToAbs
//--------------------------------------------------------------------
void CastepCellReader::fracToAbs(const Structure &st, double fx, double fy,
                                 double fz, double &cx, double &cy, double &cz) {
  // r_cart = fx*a + fy*b + fz*c
  //    a, b, c are rows of the 3×3 matrix in st.cell
  cx = fx*st.cell(0,0) + fy*st.cell(1,0) + fz*st.cell(2,0);
  cy = fx*st.cell(0,1) + fy*st.cell(1,1) + fz*st.cell(2,1);
  cz = fx*st.cell(0,2) + fy*st.cell(1,2) + fz*st.cell(2,2);
}

//--------------------------------------------------------------------
// parseSpecies
//--------------------------------------------------------------------
tadah::core::Element CastepCellReader::parseSpecies(const std::string &sym) {
  // A custom species 'Fe:up' / 'Si:1' is its element (castep_species.h).
  return castep_species_element(sym);
}
} // namespace mlip
} // namespace tadah
