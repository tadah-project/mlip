#include <tadah/mlip/structure_readers/aflow_reader.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <vector>
namespace tadah {
namespace mlip {

AflowReader::AflowReader()
  : m_dataProvider(std::make_unique<AFLOWDataProvider>()) {
}

void AflowReader::read(const std::string &auid) {
  fetchAndParseAflowByAuid(auid);
}

Structure AflowReader::getStructure() const {
  return m_structure;
}

void AflowReader::fetchAndParseAflowByAuid(const std::string &auid) {
  RawResponse rr = m_dataProvider->QueryByID(auid);
  if (rr.httpStatus != 200) {
    std::ostringstream oss;
    oss << "AflowReader: HTTP " << rr.httpStatus
        << " for AUID '" << auid << "'.";
    throw std::runtime_error(oss.str());
  }
  parseAflowJson(rr.response);
}
/*
  parseAflowJson(jsonContent):
    Interprets the final AFLOW JSON, which includes 
    lattice parameters ("geometry"), composition, species list, 
    and fractional positions.

    - The "compound" is appended to the label. 
    - "volume_cell", "density", and "auid"
      are placed in the label for clarity.
    - "geometry" is used to build a 3×3 cell matrix via makeCellMatrix()
      (a along x, b in the xy plane; row i is lattice vector a_i).
    - "positions_fractional" (relative to "geometry") plus stoichiometry &
      species become the final atoms, at r = f0 a0 + f1 a1 + f2 a2.
      "positions_cartesian" is not used: it is expressed in AFLOW's own
      frame, which in general is not the orientation built here.
    - An entry without "geometry" or "positions_fractional" is refused.

  If the species or positions do not match in count, 
  the atoms are labeled "X" as a fallback.
*/
void AflowReader::parseAflowJson(const std::string &jsonContent) {
  using json = nlohmann::json;

  json root = json::parse(jsonContent);

  //------------------------------------------------------------------
  // 1) Build a label with: compound, volume, 
  //    density, AUID.
  //------------------------------------------------------------------
  std::string compound   = root.value("compound", "UnknownCompound");
  std::string composition = root.value("composition", "N/A");
  std::string species     = root.value("species", "N/A");
  std::string volumeCell  = root.value("volume_cell", "N/A");
  std::string density     = root.value("density", "N/A");
  std::string auid        = root.value("auid", "N/A");

  std::ostringstream lbl;
  lbl << auid << " | " << compound
      << " | Volume: "     << volumeCell
      << " | Density: "    << density;
  m_structure.label = lbl.str();

  //------------------------------------------------------------------
  // 2) Extract geometry: "a,b,c,alpha,beta,gamma".
  //------------------------------------------------------------------
  double a     = 1.0;
  double b     = 1.0;
  double c     = 1.0;
  double alpha = 90.0;
  double beta  = 90.0;
  double gamma = 90.0;

  if (!root.contains("geometry")) {
    throw std::runtime_error("AflowReader: entry '" + auid +
                             "' has no \"geometry\" (a,b,c,alpha,beta,gamma).");
  }
  {
    std::string geometryStr = root["geometry"].get<std::string>();
    std::stringstream sgeom(geometryStr);
    std::vector<double> vals;
    while (sgeom.good()) {
      if (sgeom.peek() == ',') sgeom.ignore();
      double tmp;
      if (sgeom >> tmp) {
        vals.push_back(tmp);
      } else {
        sgeom.clear();
        sgeom.ignore();
      }
    }
    if (vals.size() != 6) {
      throw std::runtime_error("AflowReader: entry '" + auid +
                               "': \"geometry\" needs six numbers "
                               "(a,b,c,alpha,beta,gamma), got '" +
                               geometryStr + "'.");
    }
    a     = vals[0];
    b     = vals[1];
    c     = vals[2];
    alpha = vals[3];
    beta  = vals[4];
    gamma = vals[5];
  }

  // Create 3×3 lattice from (a, b, c, alpha, beta, gamma)
  double cellMat[3][3];
  makeCellMatrix(a, b, c, alpha, beta, gamma, cellMat);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m_structure.cell(i, j) = cellMat[i][j];
    }
  }

  //------------------------------------------------------------------
  // 3) Parse species (e.g. "Ag,K,Se,Ta"), storing each symbol in a vector.
  //------------------------------------------------------------------
  std::vector<std::string> speciesList;
  {
    std::stringstream sss(species);
    while (sss.good()) {
      if (sss.peek() == ',') sss.ignore();
      std::string sp;
      if (std::getline(sss, sp, ',')) {
        speciesList.push_back(sp);
      }
    }
  }

  //------------------------------------------------------------------
  // 4) Parse composition (e.g. "6,6,16,4") into integer stoich counts.
  //------------------------------------------------------------------
  std::vector<int> stoichCount;
  {
    std::stringstream sst(composition);
    while (sst.good()) {
      if (sst.peek() == ',') sst.ignore();
      double st;
      if (sst >> st) {
        // Round in case of fractional
        int count = static_cast<int>(std::round(st));
        stoichCount.push_back(count);
      } else {
        sst.clear();
        sst.ignore();
      }
    }
  }

  //------------------------------------------------------------------
  // 5) Acquire fractional positions from "positions_fractional":
  //       "f1,f2,f3;f1,f2,f3;..."  and convert each with the rows of the
  //    cell: r = f0*a0 + f1*a1 + f2*a2.
  //------------------------------------------------------------------
  if (!root.contains("positions_fractional")) {
    throw std::runtime_error("AflowReader: entry '" + auid +
                             "' has no \"positions_fractional\".");
  }
  std::vector<std::array<double,3>> positions;
  {
    std::string posStr = root["positions_fractional"].get<std::string>();
    std::stringstream sPos(posStr);
    while (sPos.good()) {
      std::string triplet;
      if (!std::getline(sPos, triplet, ';')) break;
      if (triplet.empty()) continue;

      std::array<double,3> coords{0,0,0};
      std::stringstream sTrip(triplet);
      for (int i = 0; i < 3; ++i) {
        if (sTrip.peek() == ',') sTrip.ignore();
        double val = 0.0;
        if (sTrip >> val) {
          coords[i] = val;
        } else {
          sTrip.clear();
          sTrip.ignore();
        }
      }
      std::array<double,3> r{0,0,0};
      for (int k = 0; k < 3; ++k) {
        r[k] = coords[0] * cellMat[0][k] + coords[1] * cellMat[1][k] +
               coords[2] * cellMat[2][k];
      }
      positions.push_back(r);
    }
  }

  //------------------------------------------------------------------
  // 6) Construct the final atoms for m_structure: 
  //    if species+composition is consistent, each site gets 
  //    the correct element symbol; otherwise, fallback "X".
  //------------------------------------------------------------------
  m_structure.atoms.clear();
  m_structure.atoms.reserve(positions.size());

  size_t totalCount = 0;
  for (auto c : stoichCount) totalCount += c;

  bool consistent = (!speciesList.empty() 
                     && !stoichCount.empty()
                     && totalCount == positions.size());

  if (!consistent) {
    // If mismatch, label everything "X".
    for (auto &coord : positions) {
      Atom atom;
      auto unknown = tadah::core::PeriodicTable().find_by_symbol("X");
      static_cast<tadah::core::Element &>(atom) = unknown;
      atom.position[0] = coord[0];
      atom.position[1] = coord[1];
      atom.position[2] = coord[2];
      m_structure.atoms.push_back(atom);
    }
    return;
  }

  // Distribute species among positions in order.
  size_t idxPos = 0;
  for (size_t iSpec = 0; iSpec < speciesList.size(); ++iSpec) {
    tadah::core::Element e = tadah::core::PeriodicTable().find_by_symbol(speciesList[iSpec]);
    for (int rep = 0; rep < stoichCount[iSpec]; ++rep) {
      Atom atom;
      static_cast<tadah::core::Element &>(atom) = e;
      atom.position[0] = positions[idxPos][0];
      atom.position[1] = positions[idxPos][1];
      atom.position[2] = positions[idxPos][2];
      m_structure.atoms.push_back(atom);
      ++idxPos;
    }
  }
}

/*
  makeCellMatrix():
    Builds the standard 3×3 lattice vectors from:
       - a,b,c (lattice lengths)
       - alpha, beta, gamma (lattice angles in degrees)

    The formula is standard:
      R0 = ( a, 0, 0 )
      R1 = ( b cos(γ),  b sin(γ),         0 )
      R2 = ( c cos(β),  c [cos(α) − cos(β) cos(γ)] / sin(γ),  
            c sqrt( 1 − cos²(β) − ... ) )

    The final line for c is chosen so the angles are satisfied.
*/
void AflowReader::makeCellMatrix(double a, double b, double c,
                                 double alphaDeg, double betaDeg, double gammaDeg,
                                 double (&cell)[3][3]) {
  double alpha = alphaDeg * M_PI / 180.0;
  double beta  = betaDeg  * M_PI / 180.0;
  double gamma = gammaDeg * M_PI / 180.0;

  cell[0][0] = a;
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;

  cell[1][0] = b * std::cos(gamma);
  cell[1][1] = b * std::sin(gamma);
  cell[1][2] = 0.0;

  cell[2][0] = c * std::cos(beta);

  cell[2][1] = c * (std::cos(alpha) - std::cos(beta)*std::cos(gamma)) 
                 / std::sin(gamma);

  double term = 1.0
                - std::cos(beta) * std::cos(beta)
                - std::pow(
                    (std::cos(alpha) 
                       - std::cos(beta)*std::cos(gamma)) 
                    / std::sin(gamma),
                    2.0
                  );
  cell[2][2] = c * std::sqrt(term > 0.0 ? term : 0.0);
  // Remaining off-diagonal elements are zero-initialized by default.
}
} // namespace mlip
} // namespace tadah
