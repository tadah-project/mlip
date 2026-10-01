#include "tadah/core/element.h"
#include "tadah/core/utils.h"
#include <string>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/castep_species.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>
namespace tadah {
namespace mlip {

CastepCastepReader::CastepCastepReader(StructureDB& db) : DatasetReader(db) {}

CastepCastepReader::CastepCastepReader(StructureDB& db, const std::string& filename) 
: DatasetReader(db, filename) {
  read_data(filename);
}

void CastepCastepReader::read_data(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    raw_data_ += line + "\n";
  }

  file.close();
}

void CastepCastepReader::parse_data() {
  // Enumerates possible parse states for line-by-line scanning
  enum class CalcState {
    Idle,               // No recognized computation
    GeometryOptimization,
    MolecularDynamics,
    SinglePointEnergy,        // SCF or single-point run
    BandStructure,
    FiniteBasisSetCorrection  // We get last SCF energy here
  };

  struct Label {
    Label(const std::string& filename) : filename(filename) {}
    CalcState calcState = CalcState::Idle;
    std::string simType = "Unknown";
    std::string algo = "Unknown";
    std::string ensemble = "Unknown";
    int startLine = -1;
    int stopLine = -1;
    int positionsStartLine = -1;
    int positionsStopLine = -1;
    int forcesStartLine = -1;
    int forcesStopLine = -1;
    int stressStartLine = -1;
    int stressStopLine = -1;
    int energyLine = -1;
    double T=-1.0;
    std::string stress="DNF";
    bool useStress=false;
    std::size_t stepNumber=0;
    std::string filename;

    std::string getStateLabel() {
      switch (calcState) {
        case CalcState::Idle: return "Idle";
        case CalcState::GeometryOptimization: return "GO";
        case CalcState::MolecularDynamics: return "MD";
        case CalcState::SinglePointEnergy: return "SPE";
        case CalcState::BandStructure: return "BS";
        case CalcState::FiniteBasisSetCorrection: return "FBSC";
      }
      return "Unknown";
    }
    std::string getMDLabel() {
      std::string state = getStateLabel();
      return "File: " + filename + " Lines: (" + std::to_string(startLine) + " : " + std::to_string(stopLine) + ") "
      + simType + " " + ensemble + " T: " + std::to_string(T) + " Step: " + std::to_string(stepNumber) + " Stress: " + stress
      + " Parser State: " + state
      + " | C: (" + std::to_string(positionsStartLine) + " : " + std::to_string(positionsStopLine)
      + ") F: (" + std::to_string(forcesStartLine) + " : " + std::to_string(forcesStopLine)
      + ") S: (" + std::to_string(stressStartLine) + " : " + std::to_string(stressStopLine) + ") + E: " + std::to_string(energyLine);
    }
    std::string getGOLabel() {
      std::string state = getStateLabel();
      return "File: " + filename + " Lines: (" + std::to_string(startLine) + " : " + std::to_string(stopLine) + ") " 
      + simType + " " + algo + " Step: " + std::to_string(stepNumber) + " stress: " + stress + " Parser State: " + state
      + " | C: (" + std::to_string(positionsStartLine) + " : " + std::to_string(positionsStopLine)
      + ") F: (" + std::to_string(forcesStartLine) + " : " + std::to_string(forcesStopLine)
      + ") S: (" + std::to_string(stressStartLine) + " : " + std::to_string(stressStopLine) + ") + E: " + std::to_string(energyLine);
    }
    std::string getSPLabel() {
      std::string state = getStateLabel();
      return "File: " + filename + " Lines: (" + std::to_string(startLine) + " : " + std::to_string(stopLine) + ") " 
      + simType + " SPE #: " + std::to_string(stepNumber) + " stress: " + stress + " Parser State: " + state
      + " | C: (" + std::to_string(positionsStartLine) + " : " + std::to_string(positionsStopLine)
      + ") F: (" + std::to_string(forcesStartLine) + " : " + std::to_string(forcesStopLine)
      + ") S: (" + std::to_string(stressStartLine) + " : " + std::to_string(stressStopLine) + ") + E: " + std::to_string(energyLine);
    }
    std::string getBSLabel() {
      std::string state = getStateLabel();
      return "File: " + filename + " Lines: (" + std::to_string(startLine) + " : " + std::to_string(stopLine) + ") " 
      + simType + " BS #: " + std::to_string(stepNumber) + " stress: " + stress + " Parser State: " + state
      + " | C: (" + std::to_string(positionsStartLine) + " : " + std::to_string(positionsStopLine)
      + ") F: (" + std::to_string(forcesStartLine) + " : " + std::to_string(forcesStopLine)
      + ") S: (" + std::to_string(stressStartLine) + " : " + std::to_string(stressStopLine) + ") + E: " + std::to_string(energyLine);
    }
    std::string getFBSCLabel() {
      std::string state = getStateLabel();
      return "File: " + filename + " Lines: (" + std::to_string(startLine) + " : " + std::to_string(stopLine) + ") " 
      + simType + " FBSC #: " + std::to_string(stepNumber) + " stress: " + stress + " Parser State: " + state
      + " | C: (" + std::to_string(positionsStartLine) + " : " + std::to_string(positionsStopLine)
      + ") F: (" + std::to_string(forcesStartLine) + " : " + std::to_string(forcesStopLine)
      + ") S: (" + std::to_string(stressStartLine) + " : " + std::to_string(stressStopLine) + ") + E: " + std::to_string(energyLine);
    }
    std::string getLabel() {
      switch (calcState) {
        case CalcState::Idle: return "Idle";
        case CalcState::GeometryOptimization: return getGOLabel();
        case CalcState::MolecularDynamics: return getMDLabel();
        case CalcState::SinglePointEnergy: return getSPLabel();
        case CalcState::BandStructure: return getBSLabel();
        case CalcState::FiniteBasisSetCorrection: return getFBSCLabel();
      }
      return "Unknown";
    }

  } simLabel(filename);

  // Current parser state and relevant flags
  CalcState current_state = CalcState::Idle; 
  bool stress_tensor_found = false;
  bool data_block_complete = false; // Indicates a completed set of structure data

  // Local counters and accumulators
  size_t natoms = 0;                // Number of atoms from the input
  size_t line_counter = 0;          // Tracks line number for warnings
  bool debug = false;               // Local debug toggle

  // Temporary container to hold ongoing parse results
  Structure s;              
  std::vector<Atom> atoms;
  // The geometry an SCF result belongs to: the last cell and positions
  // printed before it. They are taken at the SCF's first result (free
  // energy, forces or stress) when a geometry has been printed since the
  // previous result, so the last SCF of an (L)BFGS iteration -- a line or
  // quadratic minimisation step, after the trial step -- pairs with its own
  // geometry. A geometry printed after the last results -- the earlier
  // configuration a line search reverts to, before 'finished iteration' --
  // has no results and never pairs with them.
  tadah::core::Matrix3d result_cell;
  std::vector<Atom> result_atoms;
  bool result_geometry_taken = false;
  bool geometry_printed = false;  // a cell or positions block since the last result
  int result_positions_start = -1, result_positions_stop = -1;  // for the label
  auto snapshot_geometry = [&]() {
    result_cell = s.cell;
    result_atoms = atoms;
    result_positions_start = simLabel.positionsStartLine;
    result_positions_stop = simLabel.positionsStopLine;
    result_geometry_taken = true;
    geometry_printed = false;
  };
  auto take_result_geometry = [&]() {
    if (!result_geometry_taken || geometry_printed) snapshot_geometry();
  };

  std::string line;
  auto jumpNLines = [&](std::istringstream& stream, int n) {
    for (int i = 0; i < n; ++i) {
      if (!std::getline(stream, line)) {
        std::cerr << "Warning: Unexpected end of file while skipping lines" << std::endl;
        break;
      }
      ++line_counter;
    }
  };

  // Internal helper to finalize a structure's data:
  // This updates derived data (like stress if absent) and adds to the structure DB.
  auto finalize_structure = [&]() {
    // If stress tensor was not parsed, set it to zero
    if (!stress_tensor_found) {
      s.stress.set_zero();
    } else {
      s.stress *= p_conv; // Convert from GPa to eV/Å³
    }
    // We have to compute absolute positions here from temporary fractional positions
    // In case of volume-only relaxation fractional positions are reported only once
    // Rows-as-lattice-vectors: r = cell^T * frac.
    if (!result_geometry_taken) snapshot_geometry();  // a block without forces, stress or energy
    s.cell = result_cell;
    s.atoms.resize(result_atoms.size());
    for (size_t i = 0; i < result_atoms.size(); ++i) {
      // The whole element: symbol and Z (Z 0 made every species one).
      static_cast<tadah::core::Element &>(s.atoms[i]) = result_atoms[i];
      s.atoms[i].position = s.cell.transpose() * result_atoms[i].position;
      s.atoms[i].force = result_atoms[i].force;
    }
    simLabel.positionsStartLine = result_positions_start;
    simLabel.positionsStopLine = result_positions_stop;
    s.label = simLabel.getLabel();
    stdb.add(s);
    result_geometry_taken = false;
  };

  auto reset_flags = [&](CalcState new_state) {
    current_state = new_state;
    stress_tensor_found  = false;
    data_block_complete  = false;
    result_geometry_taken = false;
    if (debug) std::cout << "Resetting flags for new calculation type" << std::endl;
  };

  // Use a string stream to iterate over lines in raw_data_ 
  std::istringstream stream(raw_data_);
  while (std::getline(stream, line)) {
    ++line_counter;

    // Detect starting SP calculation (zeroth step)
    if (line.find("single point energy") != std::string::npos) {
      reset_flags(CalcState::SinglePointEnergy);
      simLabel.calcState = CalcState::SinglePointEnergy;
      simLabel.startLine = line_counter;
      simLabel.simType = "CASTEP SPE";
      simLabel.stepNumber = 0;
      if (debug) std::cout << "single point energy" << std::endl;
    }

    if (line.find("band structure") != std::string::npos) {
      reset_flags(CalcState::BandStructure);
      simLabel.calcState = CalcState::BandStructure;
      simLabel.startLine = line_counter;
      simLabel.simType = "CASTEP BS";
      simLabel.stepNumber = 0;
      if (debug) std::cout << "band structure" << std::endl;
    }

    // Detect starting GO calculation (zeroth step)
    if (line.find("geometry optimization") != std::string::npos) {
      reset_flags(CalcState::GeometryOptimization);
      simLabel.calcState = CalcState::GeometryOptimization;
      simLabel.startLine = line_counter;
      simLabel.simType = "CASTEP GO";
      simLabel.stepNumber = 0;
      if (debug) std::cout << "geometry optimization" << std::endl;
    }

    // Detect MD calculation type (zeroth step)
    if (line.find("molecular dynamics") != std::string::npos) {
      reset_flags(CalcState::MolecularDynamics);
      simLabel.calcState = CalcState::MolecularDynamics;
      simLabel.startLine = line_counter;
      simLabel.simType = "CASTEP MD";
      simLabel.stepNumber = 0;
      if (debug) std::cout << "molecular dynamics" << std::endl;
    }

    if (line.find("stress calculation") != std::string::npos) {
      std::istringstream iss(line);
      std::string ignore, stress;
      iss >> ignore >> ignore >> ignore >> stress;
      simLabel.stress = stress;
      simLabel.useStress = stress == "on" ? true : false;
      if (debug) std::cout << "stress calculation" << std::endl;
    }

    // Get GO algorithm name at step zero
    if (line.find("Geometry Optimization Parameters") != std::string::npos) {
      std::string ignore, algo;
      jumpNLines(stream, 2);
      std::istringstream iss(line);
      iss >> ignore >> ignore >> ignore >> algo;
      simLabel.algo = algo;
      if (debug) std::cout << "Geometry Optimization Parameters" << std::endl;
    }

    // Get MD ensemble name at step zero
    if (line.find("Molecular Dynamics Parameters") != std::string::npos) {
      reset_flags(CalcState::MolecularDynamics);
      jumpNLines(stream, 2);
      std::istringstream iss(line);
      std::string ignore, ensemble;
      iss >> ignore >> ignore >> ensemble;
      simLabel.ensemble = ensemble;
      if (debug) std::cout << "Molecular Dynamics Parameters" << std::endl;
    }

    // Detect starting of basis set correction calculation (FBSC)
    if (line.find("Calculating finite basis set correction") != std::string::npos) {
      reset_flags(CalcState::FiniteBasisSetCorrection);
      simLabel.calcState = CalcState::FiniteBasisSetCorrection;
      simLabel.startLine = line_counter;
      simLabel.simType = "CASTEP FBSC";
      simLabel.stepNumber = 0;
      if (debug) std::cout << "Basis set correction" << std::endl;
    }


    // Detect or store the "Unit Cell"
    if (line.find("Unit Cell") != std::string::npos) {
      // A new cell is encountered. Prepare to read the next three lines of vectors.
      // Finalize any incomplete data set if logic demands it, but not strictly necessary here
      if (debug) std::cout << "Reading 3 lines of Unit Cell" << std::endl;
      
      jumpNLines(stream, 2);

      // Read the 3 lattice vectors. Each row of the .castep file is a lattice
      // vector a_i; store as row i of cell.
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Unexpected end of file while reading lattice vectors at row "
                    << i << " (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;
        std::istringstream iss(line);
        if (!(iss >> s.cell(i, 0) >> s.cell(i, 1) >> s.cell(i, 2))) {
          std::cerr << "Warning: Failed to parse lattice vector at row "
                    << i << " (line: " << line_counter << ")." << std::endl;
          break;
        }
      }
      geometry_printed = true;
      continue;
    }

    // Detect total number of ions 
    if (line.find("Total number of ions in cell") != std::string::npos) {
      // Parse the integer natoms from this line
      std::istringstream iss(line);
      std::string tmp;
      int word_count = 0;
      while (iss >> tmp) {
        ++word_count;
        if (word_count == 7) {
          if (!(iss >> natoms)) {
            std::cerr << "Warning: Could not parse number of atoms in line: " << line_counter << std::endl;
          }
          break;
        }
      }
      if (debug) std::cout << "Total number of ions in cell: " << natoms << std::endl;
      continue;
    }

    // Detect GO iteration
    if (line.find("Starting BFGS iteration") != std::string::npos || 
        line.find("Starting LBFGS iteration") != std::string::npos || 
        line.find("Starting Delocalized iteration") != std::string::npos || 
        line.find("Starting DampedMD iteration") != std::string::npos || 
        line.find("Starting TPSD iteration") != std::string::npos) {
      reset_flags(CalcState::GeometryOptimization);
      std::istringstream iss(line);
      std::string ignore, algo, step_str;
      iss >> ignore >> algo >> ignore >> step_str;
      simLabel.simType = "CASTEP GO";
      simLabel.startLine = line_counter;
      simLabel.algo = algo;
      simLabel.stepNumber = std::stoi(step_str);
      simLabel.calcState = CalcState::GeometryOptimization;
      if (debug) std::cout << "Starting" +algo+ " iteration " << step_str << std::endl;
    }

    // Detect MD iteration
    if (line.find("Starting MD iteration") != std::string::npos) {
      reset_flags(CalcState::MolecularDynamics);
      std::istringstream iss(line);
      std::string ignore, step;
      iss >> ignore >> ignore >> ignore >> step;
      simLabel.simType = "CASTEP MD";
      simLabel.startLine = line_counter;
      simLabel.stepNumber = std::stoi(step);
      simLabel.calcState = CalcState::MolecularDynamics;
      if (debug) std::cout << "Starting MD iteration " << step << std::endl;
      continue;
    }

    // Detect fractional coordinates block
    if (line.find("Fractional coordinates of atoms") != std::string::npos) {
      if (debug) std::cout << "Fractional coordinates of atoms" << std::endl;
      jumpNLines(stream, 2);
      atoms.clear(); // Clear old atoms or partial data
      atoms.reserve(natoms);
      simLabel.positionsStartLine = line_counter + 1;
      for (size_t i = 0; i < natoms; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Truncated file reading fractional coords at row " << i 
                    << " (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;

        // Format is typically: " X ??? ??? frac_x frac_y frac_z "
        std::istringstream iss(line);
        std::string discard, species;
        double px, py, pz;
        if (!(iss >> discard >> species >> discard >> px >> py >> pz)) {
          std::cerr << "Warning: Unexpected parse error reading fractional coords (line: "
                    << line_counter << ")." << std::endl;
          break;
        }
        // Create Atom, convert fractional to absolute using s.cell
        Atom a(castep_species_element(species), px, py, pz, 0.0, 0.0, 0.0);
        atoms.push_back(a); // keep it fractional for now
      }
      simLabel.positionsStopLine = line_counter;
      geometry_printed = true;
    }

    // Detect forces block
    if (line.find("Cartesian components (eV/A)") != std::string::npos) {
      if (debug) std::cout << "Reading forces..." << std::endl;
      take_result_geometry();
      // Usually there are three lines of header/blank text before actual force lines
      jumpNLines(stream, 3);
      // Read force for each atom
      simLabel.forcesStartLine = line_counter+1;
      for (size_t i = 0; i < natoms; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Truncated file reading forces at row " << i
                    << " (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;

        std::istringstream iss(line);
        std::string tmp;
        double fx, fy, fz;
        if (!(iss >> tmp >> tmp >> tmp >> fx >> fy >> fz)) {
          std::cerr << "Warning: Could not parse atomic forces (line: " 
                    << line_counter << ")." << std::endl;
          break;
        }
        if (i < result_atoms.size()) {
          result_atoms[i].force = tadah::core::Vec3d(fx, fy, fz);
        }
      }
      simLabel.forcesStopLine = line_counter;

      if (simLabel.useStress == false && (current_state == CalcState::SinglePointEnergy || 
            current_state == CalcState::FiniteBasisSetCorrection ||
            current_state == CalcState::BandStructure)) {
        data_block_complete = true;
        simLabel.stopLine = line_counter;
      }
    }

    // Detect stress block
    if (line.find("Cartesian components (GPa)") != std::string::npos) {
      if (debug) std::cout << "Reading stress..." << std::endl;
      take_result_geometry();
      jumpNLines(stream, 3);
      // A 3x3 matrix is expected
      simLabel.stressStartLine = line_counter + 1;
      for (int row = 0; row < 3; ++row) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Truncated file reading stress row " << row 
                    << " (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;
        std::istringstream iss(line);
        std::string ignore;
        double v1, v2, v3;
        if (!(iss >> ignore >> ignore >> v1 >> v2 >> v3)) {
          std::cerr << "Warning: Could not parse stress row (line: " 
                    << line_counter << ")." << std::endl;
          break;
        }
        s.stress(row, 0) = v1;
        s.stress(row, 1) = v2;
        s.stress(row, 2) = v3;
        simLabel.stressStopLine = line_counter;
      }
      stress_tensor_found = true;
      if (current_state == CalcState::SinglePointEnergy || 
          current_state == CalcState::FiniteBasisSetCorrection ||
          current_state == CalcState::BandStructure) {
        data_block_complete = true;
        simLabel.stopLine = line_counter;
      }
    }

    // Detect free energy for SPE/GO/FBSC/BS
    // The energy might not be final so we need to complete the data block elsewhere.
    if (line.find("Final free energy (E-TS)") != std::string::npos && 
        (current_state == CalcState::GeometryOptimization
         || current_state == CalcState::SinglePointEnergy
         || current_state == CalcState::BandStructure
         || current_state == CalcState::FiniteBasisSetCorrection)) {
      if (debug) std::cout << "Final free energy (E-TS)" << std::endl;
      take_result_geometry();
      std::istringstream iss(line);
      std::string ignore;
      if (!(iss >> ignore >> ignore >> ignore >> ignore >> ignore >> s.energy)) {
        std::cerr << "Warning: Could not parse final free energy (line: "
          << line_counter << ")." << std::endl;
      }
      simLabel.energyLine = line_counter;
    }

    // Terminates the geometry optimization step
    // Uses last free energy found previously. Iteration 0 with the finite
    // basis set correction is in the FBSC state; it is completed by its
    // stress block, or here when none is printed (a fixed-cell run).
    const bool fbsc_pending =
        current_state == CalcState::FiniteBasisSetCorrection && result_geometry_taken;
    if (line.find("enthalpy=") != std::string::npos &&
        (current_state == CalcState::GeometryOptimization || fbsc_pending)) {
      if (debug) std::cout << "enthalpy=" << std::endl;
      // This indicates a converged or final iteration of geometry optimization
      std::istringstream iss(line);
      std::string ignore, step;
      if (!(iss >> ignore >> ignore >> ignore>> step >> ignore >> ignore >> ignore)) {
        std::cerr << "Warning: Could not parse GO termination line (line: "
                  << line_counter << ")." << std::endl;
      }
      simLabel.stepNumber = std::stoi(step);
      simLabel.stopLine = line_counter;
      data_block_complete = true;
    }

    // Detect potential energy line used in MD 
    if (line.find("Potential Energy:") != std::string::npos && current_state == CalcState::MolecularDynamics) {
      if (debug) std::cout << "Potential Energy:" << std::endl;
      std::istringstream iss(line);
      std::string ignore;
      if (!(iss >> ignore >> ignore >> ignore >> s.energy)) {
        std::cerr << "Warning: Could not parse potential energy in MD (line: " 
                  << line_counter << ")." << std::endl;
      }
      simLabel.energyLine = line_counter;
    }

    // Detect temperature line in MD and finalize
    if (line.find("Temperature:") != std::string::npos && current_state == CalcState::MolecularDynamics) {
      if (debug) std::cout << "Temperature:" << std::endl;
      std::istringstream iss(line);
      std::string ignore;
      if (!(iss >> ignore >> ignore >> s.T)) {
        std::cerr << "Warning: Could not parse temperature in MD (line: " 
                  << line_counter << ")." << std::endl;
      }
      simLabel.T = s.T;
      simLabel.stopLine = line_counter;
      data_block_complete = true;
    }

    // If a completed data set is detected, finalize
    if (data_block_complete) {
      finalize_structure();
      reset_flags(CalcState::Idle);
    }
  } // end while getline
}

void CastepCastepReader::print_summary() const {
  std::cout << get_summary();
}

std::string CastepCastepReader::get_summary() const {
  return stdb.summary();
}
} // namespace mlip
} // namespace tadah
