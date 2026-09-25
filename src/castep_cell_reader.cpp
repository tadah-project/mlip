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
    throw std::runtime_error("CastepCellReader: Cannot open file " + path);
  }

  std::ostringstream buffer;
  buffer << ifs.rdbuf();
  parseCellContents(buffer.str());

  m_structure.label = "CASTEP cell file: " + path;
}

Structure CastepCellReader::getStructure() const {
  return m_structure;
}

//--------------------------------------------------------------------
// parseCellContents
//--------------------------------------------------------------------
void CastepCellReader::parseCellContents(const std::string &contents) {
  std::istringstream iss(contents);
  std::string line;

  // State flags for each block
  bool inLatticeCart   = false;
  bool inLatticeABC    = false;
  bool inPosFracBlock  = false;
  bool inPosAbsBlock   = false;

  int  lcCount = 0; // track lines read for LATTICE_CART
  std::vector<std::string> abcLines; // store lines for LATTICE_ABC

  // Initialize cell to zero (in case we never find any block)
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      m_structure.cell(i,j) = 0.0;

  // Read file line by line
  while (std::getline(iss, line)) {
    // Make a lowercase copy for easier detection
    std::string lcase = line;
    std::transform(lcase.begin(), lcase.end(), lcase.begin(), ::tolower);

    // Detect block starts/ends
    if (lcase.find("%block lattice_cart") != std::string::npos) {
      inLatticeCart = true;
      inLatticeABC  = false;
      lcCount = 0;
      continue;
    }
    if (lcase.find("%endblock lattice_cart") != std::string::npos) {
      inLatticeCart = false;
      continue;
    }
    if (lcase.find("%block lattice_abc") != std::string::npos) {
      inLatticeABC  = true;
      inLatticeCart = false;
      lcCount = 0;
      abcLines.clear();
      continue;
    }
    if (lcase.find("%endblock lattice_abc") != std::string::npos) {
      inLatticeABC = false;
      // parse lines from abcLines (if present)
      if (abcLines.size() >= 2) {
        parseLatticeABC(abcLines);
      }
      continue;
    }
    if (lcase.find("%block positions_frac") != std::string::npos) {
      inPosFracBlock = true;
      inPosAbsBlock  = false;
      continue;
    }
    if (lcase.find("%endblock positions_frac") != std::string::npos) {
      inPosFracBlock = false;
      continue;
    }
    if (lcase.find("%block positions_abs") != std::string::npos) {
      inPosAbsBlock  = true;
      inPosFracBlock = false;
      continue;
    }
    if (lcase.find("%endblock positions_abs") != std::string::npos) {
      inPosAbsBlock = false;
      continue;
    }

    // Now parse lines if in any recognized block
    if (inLatticeCart) {
      // read up to 3 lines for a,b,c
      if (lcCount < 4) {
        auto toks = split_line(line);
        if (toks.size() == 1) {
          // skip optional units line
          continue;
        }
        if (toks.size() < 3) {
          throw std::runtime_error("CastepCellReader: LATTICE_CART line missing coords.");
        }
        double x = tadah::core::to_double(toks[0]);
        double y = tadah::core::to_double(toks[1]);
        double z = tadah::core::to_double(toks[2]);

        m_structure.cell(lcCount,0) = x;
        m_structure.cell(lcCount,1) = y;
        m_structure.cell(lcCount,2) = z;
        lcCount++;
      }
    }
    else if (inLatticeABC) {
      // store lines for later parsing
      if (!line.empty()) {
        abcLines.push_back(line);
      }
    }
    else if (inPosFracBlock) {
      // e.g. "Si 0.0 0.0 0.0"
      auto toks = split_line(line);
      if (toks.size() < 4) {
        // skip empty lines which should not be there...
        continue;
      }

      // parse species
      tadah::core::Element e = parseSpecies(toks[0]);

      // fractional coords
      double fx = tadah::core::to_double(toks[1]);
      double fy = tadah::core::to_double(toks[2]);
      double fz = tadah::core::to_double(toks[3]);

      double cx, cy, cz;
      fracToAbs(fx, fy, fz, cx, cy, cz);

      Atom a;  
      // Copy base element data into 'a'
      static_cast<tadah::core::Element&>(a) = e;
      // Now set the position
      a.position = {cx, cy, cz};

      // Force is left default (0,0,0)
      m_structure.atoms.push_back(a);
    }
    else if (inPosAbsBlock) {
      auto toks = split_line(line);
      if (toks.size() < 4) {
        // skip units line
        continue;
      }

      tadah::core::Element e = parseSpecies(toks[0]);

      double x = tadah::core::to_double(toks[1]);
      double y = tadah::core::to_double(toks[2]);
      double z = tadah::core::to_double(toks[3]);

      Atom a;
      static_cast<tadah::core::Element&>(a) = e;
      a.position = {x, y, z};

      m_structure.atoms.push_back(a);
    }
  }
}

//--------------------------------------------------------------------
// parseLatticeABC
//--------------------------------------------------------------------
void CastepCellReader::parseLatticeABC(const std::vector<std::string> &lines) {
  // We expect at least two lines:
  //   line 1: "a b c"
  //   line 2: "alpha beta gamma"
  // Additional lines might exist (units, comments).
  // We'll search for the first line with >=3 tokens for (a,b,c),
  // and the next with >=3 tokens for (alpha,beta,gamma).

  double aVal=0.0, bVal=0.0, cVal=0.0;
  double alphaDeg=90.0, betaDeg=90.0, gammaDeg=90.0;
  bool haveABC = false;
  bool haveAngles = false;

  for (auto &ln : lines) {
    auto toks = split_line(ln);
    if (toks.size() >= 3 && !haveABC) {
      aVal = tadah::core::to_double(toks[0]);
      bVal = tadah::core::to_double(toks[1]);
      cVal = tadah::core::to_double(toks[2]);
      haveABC = true;
      continue;
    }
    else if (toks.size() >= 3 && haveABC && !haveAngles) {
      alphaDeg = tadah::core::to_double(toks[0]);
      betaDeg  = tadah::core::to_double(toks[1]);
      gammaDeg = tadah::core::to_double(toks[2]);
      haveAngles = true;
      break;
    }
  }
  if (!haveABC || !haveAngles) {
    throw std::runtime_error("CastepCellReader: LATTICE_ABC block incomplete (need a,b,c and alpha,beta,gamma).");
  }

  double alpha = deg2rad(alphaDeg);
  double beta  = deg2rad(betaDeg);
  double gamma = deg2rad(gammaDeg);

  // Standard orientation:
  //   a = (aVal, 0, 0)
  //   b.x = bVal * cos(gamma)
  //   b.y = bVal * sin(gamma)
  //   b.z = 0
  //   c.x = cVal * cos(beta)
  //   c.y = cVal * [cos(alpha) - cos(beta)*cos(gamma)] / sin(gamma)
  //   c.z = sqrt(cVal^2 - cx^2 - cy^2)

  // a
  m_structure.cell(0,0) = aVal;
  m_structure.cell(0,1) = 0.0;
  m_structure.cell(0,2) = 0.0;

  // b
  double bx = bVal * std::cos(gamma);
  double by = bVal * std::sin(gamma);
  m_structure.cell(1,0) = bx;
  m_structure.cell(1,1) = by;
  m_structure.cell(1,2) = 0.0;

  // c
  double cx = cVal * std::cos(beta);
  double cy = cVal * (std::cos(alpha) - std::cos(beta)*std::cos(gamma)) / std::sin(gamma);
  double czsq = cVal*cVal - cx*cx - cy*cy;
  if (czsq < 0.0) {
    // could be numerical rounding or truly invalid
    if (czsq > -1e-10) czsq = 0.0; // clamp small negative to zero
    else {
      throw std::runtime_error("CastepCellReader: invalid angles produce negative c.z^2.");
    }
  }
  double cz = std::sqrt(czsq);

  m_structure.cell(2,0) = cx;
  m_structure.cell(2,1) = cy;
  m_structure.cell(2,2) = cz;
}

//--------------------------------------------------------------------
// fracToAbs
//--------------------------------------------------------------------
void CastepCellReader::fracToAbs(double fx, double fy, double fz, 
                                       double &cx, double &cy, double &cz) {
  // r_cart = fx*a + fy*b + fz*c
  //    a, b, c are rows of the 3×3 matrix in m_structure.cell
  cx = fx*m_structure.cell(0,0) + fy*m_structure.cell(1,0) + fz*m_structure.cell(2,0);
  cy = fx*m_structure.cell(0,1) + fy*m_structure.cell(1,1) + fz*m_structure.cell(2,1);
  cz = fx*m_structure.cell(0,2) + fy*m_structure.cell(1,2) + fz*m_structure.cell(2,2);
}

//--------------------------------------------------------------------
// parseSpecies
//--------------------------------------------------------------------
tadah::core::Element CastepCellReader::parseSpecies(const std::string &sym) {
  return tadah::core::PeriodicTable().find_by_symbol(sym);
}
} // namespace mlip
} // namespace tadah
