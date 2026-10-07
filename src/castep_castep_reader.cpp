#include "tadah/core/element.h"
#include "tadah/core/utils.h"
#include <tadah/core/number_format.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/castep_species.h>
#include <tadah/mlip/dataset_readers/missing_stress.h>
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
    throw std::runtime_error("CASTEP .castep " + filename + ": cannot open the file");
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
  bool md_task = false;  // the run's 'type of calculation' is molecular dynamics
  bool go_task = false;  // ... geometry optimization
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
  // The results of one SCF: a structure gets only its own SCF's energy,
  // forces and stress, and its label names only its own blocks. The energy
  // lines of an SCF, as printed (unset: NaN):
  //  - smeared: 'Final energy, E' and 'Final free energy (E-TS)' (F);
  //  - fixed occupancy: 'Final energy =' (no smearing: E is F);
  //  - with a dispersion correction (DFT-D, SEDC): 'Dispersion corrected
  //    final energy*' (E + correction) and, smeared, 'Dispersion corrected
  //    final free energy*';
  //  - MD: the MD Data 'Potential Energy:'.
  const double unset = std::numeric_limits<double>::quiet_NaN();
  double e_smeared = unset, e_free = unset, e_fixed = unset, e_disp = unset,
         e_disp_free = unset, e_md = unset;
  bool forces_found = false;
  auto clear_results = [&]() {
    s.energy = 0.0;
    s.stress.set_zero();
    e_smeared = e_free = e_fixed = e_disp = e_disp_free = e_md = unset;
    stress_tensor_found = false;
    forces_found = false;
    simLabel.forcesStartLine = simLabel.forcesStopLine = -1;
    simLabel.stressStartLine = simLabel.stressStopLine = -1;
    simLabel.energyLine = -1;
  };
  // The energy whose gradient the SCF's forces and stress are, or NaN.
  auto scf_energy = [&]() -> double {
    if (!std::isnan(e_md)) return e_md;
    if (!std::isnan(e_disp_free)) return e_disp_free;
    if (!std::isnan(e_free)) {
      if (std::isnan(e_disp)) return e_free;
      if (std::isnan(e_smeared))
        throw std::runtime_error(
            "CASTEP .castep " + simLabel.filename + ": line " +
            std::to_string(simLabel.energyLine) +
            ": a smeared SCF with a dispersion correction prints neither "
            "'Dispersion corrected final free energy*' nor 'Final energy, E', "
            "so its corrected free energy cannot be formed");
      return e_free + (e_disp - e_smeared);  // F plus the correction
    }
    if (!std::isnan(e_disp)) return e_disp;
    return e_fixed;
  };
  // A new geometry starts a new SCF: the results so far belong to the
  // previous one (a trial step whose line minimisation followed).
  auto snapshot_geometry = [&]() {
    if (result_geometry_taken) clear_results();
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

  // The last structure stored, for the stress block that CASTEP prints for
  // the final configuration of a geometry optimisation (after
  // '<algorithm>: Final Configuration'), which belongs to it when the
  // geometry is the same and it has no stress of its own.
  std::size_t stored = 0, stored_without_stress = 0, stored_without_forces = 0;
  bool final_configuration = false;
  bool last_has_stress = true, last_has_forces = true;
  std::string banner_title;  // the title of the last ' *** <title> ***' banner
  // An MD step is complete once its MD Data (potential energy, temperature)
  // is read; it is stored at the next step boundary, so that a stress block
  // CASTEP prints after the MD Data (properties sampled every N steps) is
  // the step's.
  bool md_step_ready = false;
  tadah::core::Matrix3d last_cell;
  std::vector<Atom> last_atoms;
  auto same_geometry_as_last = [&]() {
    // Exactly the same printed numbers.
    if (!s.cell.isApprox(last_cell, 0.0) || atoms.size() != last_atoms.size()) return false;
    for (size_t i = 0; i < atoms.size(); ++i)
      if (!atoms[i].position.isApprox(last_atoms[i].position, 0.0) ||
          atoms[i].symbol != last_atoms[i].symbol)
        return false;
    return true;
  };

  std::string line;
  // Errors name the file and the line. A value that cannot be read refuses
  // the file; a forces or stress line that cannot be read (a value CASTEP
  // could not print, '*****') is no forces or stress for that structure,
  // with a warning; a block the file ends inside (a run stopped while
  // writing it) warns once.
  auto refuse = [&](const std::string &what) {
    throw std::runtime_error("CASTEP .castep " + simLabel.filename + ": line " +
                             std::to_string(line_counter) + ": " + what);
  };
  auto quoted = [&]() { return "'" + tadah::core::trim(line) + "'"; };
  auto warn = [&](const std::string &what) {
    std::cerr << "WARNING: CASTEP .castep " << simLabel.filename << ": line "
              << line_counter << ": " << what << std::endl;
  };
  bool ended_inside = false;  // the file ends inside a block (warned)
  auto ends_inside = [&](const std::string &block) {
    if (!ended_inside)
      std::cerr << "WARNING: CASTEP .castep " << simLabel.filename
                << ": the file ends inside the " << block << " block at line "
                << line_counter << std::endl;
    ended_inside = true;
  };
  // An iteration or step number: the whole token, an integer.
  auto step_number = [&](const std::string &token) -> std::size_t {
    if (!tadah::core::is_integer_literal(token) || token[0] == '-')
      refuse("cannot read the iteration number from " + quoted());
    try {
      return static_cast<std::size_t>(std::stoull(token));
    } catch (const std::exception &) {
      refuse("cannot read the iteration number from " + quoted());
    }
    return 0;
  };
  auto jumpNLines = [&](std::istringstream& stream, int n,
                        const std::string &block) {
    for (int i = 0; i < n; ++i) {
      if (!std::getline(stream, line)) {
        ends_inside(block);
        return false;
      }
      ++line_counter;
    }
    return true;
  };

  // Internal helper to finalize a structure's data:
  // This updates derived data (like stress if absent) and adds to the structure DB.
  auto finalize_structure = [&]() {
    if (!result_geometry_taken) snapshot_geometry();  // a block without forces, stress or energy
    const double energy = scf_energy();
    if (std::isnan(energy)) {
      // A band structure continued from a previous run does no SCF: there
      // is no structure to store.
      if (current_state == CalcState::BandStructure) {
        std::cerr << "WARNING: CASTEP .castep " << simLabel.filename
                  << ": the band-structure step ending at line " << line_counter
                  << " has no SCF energy (a continuation run); no structure "
                     "is stored for it."
                  << std::endl;
        result_geometry_taken = false;
        clear_results();
        return;
      }
      throw std::runtime_error(
          "CASTEP .castep " + simLabel.filename + ": structure " +
          std::to_string(stored + 1) + " (the " + simLabel.getStateLabel() +
          " step ending at line " + std::to_string(line_counter) +
          ") has no energy of its own: no 'Final free energy (E-TS)', "
          "'Final energy =' or MD 'Potential Energy:' line follows its "
          "geometry");
    }
    s.energy = energy;
    // A structure without forces of its own is stored with zero forces and
    // force weight 0, so that a FORCE fit ignores them.
    if (!forces_found) {
      for (auto &a : result_atoms) a.force = tadah::core::Vec3d(0.0, 0.0, 0.0);
      s.fweight = 0.0;
      ++stored_without_forces;
    } else {
      s.fweight = 1.0;
    }
    // A structure without a stress tensor of its own is stored with zero
    // stress and stress weight 0, so that a STRESS fit ignores it.
    if (!stress_tensor_found) {
      s.stress.set_zero();
      s.sweight = 0.0;
      ++stored_without_stress;
    } else {
      s.stress *= p_conv; // Convert from GPa to eV/Å³
      s.sweight = 1.0;
    }
    // We have to compute absolute positions here from temporary fractional positions
    // In case of volume-only relaxation fractional positions are reported only once
    // Rows-as-lattice-vectors: r = cell^T * frac.
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
    ++stored;
    last_has_stress = stress_tensor_found;
    last_has_forces = forces_found;
    last_cell = result_cell;
    last_atoms = result_atoms;
    final_configuration = false;
    result_geometry_taken = false;
    clear_results();
    s.T = 0.0;
  };

  auto reset_flags = [&](CalcState new_state) {
    current_state = new_state;
    data_block_complete  = false;
    result_geometry_taken = false;
    clear_results();
    if (debug) std::cout << "Resetting flags for new calculation type" << std::endl;
  };

  // Use a string stream to iterate over lines in raw_data_ 
  std::istringstream stream(raw_data_);
  while (std::getline(stream, line)) {
    ++line_counter;

    // The end of an MD step: the next step, its '... finished MD iteration'
    // line, the end of the run or the next run.
    if (md_step_ready &&
        (line.find("finished MD iteration") != std::string::npos ||
         line.find("Starting MD iteration") != std::string::npos ||
         line.find("Finished MD") != std::string::npos ||
         line.find("type of calculation") != std::string::npos)) {
      finalize_structure();
      md_step_ready = false;
      reset_flags(CalcState::Idle);
    }

    // A run starts with its 'type of calculation' line: the run type comes
    // from that line only, and nothing of an earlier run carries over.
    if (line.find("type of calculation") != std::string::npos &&
        line.find(':') != std::string::npos) {
      const std::string task = tadah::core::trim(line.substr(line.find(':') + 1));
      final_configuration = false;
      md_task = task == "molecular dynamics";
      go_task = task == "geometry optimization";
      CalcState state = CalcState::Idle;
      std::string type = "CASTEP";
      if (task == "single point energy") {
        state = CalcState::SinglePointEnergy;
        type = "CASTEP SPE";
      } else if (task == "band structure") {
        state = CalcState::BandStructure;
        type = "CASTEP BS";
      } else if (go_task) {
        state = CalcState::GeometryOptimization;
        type = "CASTEP GO";
      } else if (md_task) {
        state = CalcState::MolecularDynamics;
        type = "CASTEP MD";
      }
      // Any other run type (phonons, spectroscopy, ...) stores nothing.
      reset_flags(state);
      simLabel.calcState = state;
      simLabel.startLine = line_counter;
      simLabel.simType = type;
      simLabel.stepNumber = 0;
      if (debug) std::cout << "type of calculation: " << task << std::endl;
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
      if (!jumpNLines(stream, 2, "Geometry Optimization Parameters")) break;
      std::istringstream iss(line);
      iss >> ignore >> ignore >> ignore >> algo;
      simLabel.algo = algo;
      if (debug) std::cout << "Geometry Optimization Parameters" << std::endl;
    }

    // Get MD ensemble name at step zero
    if (line.find("Molecular Dynamics Parameters") != std::string::npos) {
      reset_flags(CalcState::MolecularDynamics);
      if (!jumpNLines(stream, 2, "Molecular Dynamics Parameters")) break;
      std::istringstream iss(line);
      std::string ignore, ensemble;
      iss >> ignore >> ignore >> ensemble;
      simLabel.ensemble = ensemble;
      if (debug) std::cout << "Molecular Dynamics Parameters" << std::endl;
    }

    // Detect starting of basis set correction calculation (FBSC). In an MD
    // run the correction's SCFs belong to step 0, which is completed at its
    // MD Data lines like every later step.
    // A run of another type (phonons, spectroscopy, ...) stores nothing,
    // its correction included.
    if (line.find("Calculating finite basis set correction") != std::string::npos &&
        (current_state == CalcState::SinglePointEnergy ||
         current_state == CalcState::BandStructure ||
         current_state == CalcState::GeometryOptimization)) {
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
      
      if (!jumpNLines(stream, 2, "Unit Cell")) break;

      // Read the 3 lattice vectors. Each row of the .castep file is a lattice
      // vector a_i; store as row i of cell.
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          ends_inside("Unit Cell");
          break;
        }
        ++line_counter;
        std::istringstream iss(line);
        if (!(iss >> s.cell(i, 0) >> s.cell(i, 1) >> s.cell(i, 2)))
          refuse("lattice vector " + std::to_string(i + 1) +
                 " of the Unit Cell block, " + quoted() +
                 ", is not three numbers");
      }
      if (ended_inside) break;
      geometry_printed = true;
      continue;
    }

    // Detect total number of ions 
    if (line.find("Total number of ions in cell") != std::string::npos) {
      // Parse the integer natoms from this line
      // 'Total number of ions in cell = <n>'
      std::istringstream iss(line.substr(line.find('=') + 1));
      std::string count, rest;
      // A count: digits, at most 10^12 (an absurd count is not a count).
      if (!(iss >> count) || (iss >> rest) || count.size() > 13 ||
          !tadah::core::is_integer_literal(count) || count[0] == '-' ||
          std::stoull(count) > 1000000000000ULL)
        refuse("cannot read the number of ions from " + quoted());
      natoms = static_cast<size_t>(std::stoull(count));
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
      simLabel.stepNumber = step_number(step_str);
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
      simLabel.stepNumber = step_number(step);
      simLabel.calcState = CalcState::MolecularDynamics;
      if (debug) std::cout << "Starting MD iteration " << step << std::endl;
      continue;
    }

    // Detect fractional coordinates block
    if (line.find("Fractional coordinates of atoms") != std::string::npos) {
      if (debug) std::cout << "Fractional coordinates of atoms" << std::endl;
      if (!jumpNLines(stream, 2, "fractional coordinates")) break;
      atoms.clear(); // Clear old atoms or partial data
      atoms.reserve(natoms);
      simLabel.positionsStartLine = line_counter + 1;
      for (size_t i = 0; i < natoms; ++i) {
        if (!std::getline(stream, line)) {
          ends_inside("fractional coordinates");
          break;
        }
        ++line_counter;

        // Format is typically: " X ??? ??? frac_x frac_y frac_z "
        std::istringstream iss(line);
        std::string discard, species;
        double px, py, pz;
        if (!(iss >> discard >> species >> discard >> px >> py >> pz))
          refuse("the position of atom " + std::to_string(i + 1) + ", " +
                 quoted() + ", is not 'x <species> <number> u v w x'");
        // Create Atom, convert fractional to absolute using s.cell
        tadah::core::Element element;
        try {
          element = castep_species_element(species);
        } catch (const std::exception &e) {
          refuse(e.what());
        }
        Atom a(element, px, py, pz, 0.0, 0.0, 0.0);
        atoms.push_back(a); // keep it fractional for now
      }
      if (ended_inside) break;
      simLabel.positionsStopLine = line_counter;
      geometry_printed = true;
    }

    // The title of every boxed ' *** <title> ***' banner. Only the physical
    // forces and stress are read: 'Forces', 'Symmetrised Forces',
    // 'Unconstrained Forces', 'Stress Tensor', 'Symmetrised Stress Tensor'.
    // Not the projections a constrained optimisation uses ('Constrained
    // ...'), the per-term breakdown printed at iprint >= 2 ('Ewald forces',
    // 'Local potential forces', ...) or MD's 'Finite Temperature Pressure
    // Tensor'.
    if (line.rfind(" ***", 0) == 0) {
      std::string title = line;
      title.erase(std::remove(title.begin(), title.end(), '*'), title.end());
      title = tadah::core::trim(title);
      if (!title.empty()) banner_title = title;
    }
    const bool physical_forces = banner_title == "Forces" ||
                                 banner_title == "Symmetrised Forces" ||
                                 banner_title == "Unconstrained Forces";
    const bool physical_stress = banner_title == "Stress Tensor" ||
                                 banner_title == "Symmetrised Stress Tensor";

    // Detect forces block
    if (line.find("Cartesian components (eV/A)") != std::string::npos &&
        physical_forces) {
      if (debug) std::cout << "Reading forces..." << std::endl;
      take_result_geometry();
      // Usually there are three lines of header/blank text before actual force lines
      if (!jumpNLines(stream, 3, "forces")) break;
      // Read force for each atom
      simLabel.forcesStartLine = line_counter+1;
      for (size_t i = 0; i < natoms; ++i) {
        if (!std::getline(stream, line)) {
          ends_inside("forces");
          break;
        }
        ++line_counter;

        std::istringstream iss(line);
        std::string tmp;
        double fx, fy, fz;
        if (!(iss >> tmp >> tmp >> tmp >> fx >> fy >> fz)) {
          warn("the forces line of atom " + std::to_string(i + 1) + ", " +
               quoted() + ", cannot be read; the structure is stored without "
               "forces");
          break;
        }
        if (i < result_atoms.size()) {
          result_atoms[i].force = tadah::core::Vec3d(fx, fy, fz);
        }
        if (i + 1 == natoms) forces_found = true;
      }
      simLabel.forcesStopLine = line_counter;

      // Iteration 0 of a geometry optimisation (the FBSC state) is completed
      // at 'enthalpy=', after its stress, which a variable-cell run prints
      // whatever 'stress calculation' says.
      if (simLabel.useStress == false && (current_state == CalcState::SinglePointEnergy ||
            (current_state == CalcState::FiniteBasisSetCorrection && !go_task) ||
            current_state == CalcState::BandStructure)) {
        data_block_complete = true;
        simLabel.stopLine = line_counter;
      }
      // The final configuration's forces (a run whose iterations print none,
      // e.g. with every ion fixed): they belong to the last iteration when
      // the geometry is the same and it has no forces of its own.
      if (current_state == CalcState::Idle && final_configuration && stored > 0) {
        if (forces_found && !last_has_forces && same_geometry_as_last()) {
          Structure &last = stdb(stdb.size() - 1);
          for (size_t i = 0; i < last.atoms.size() && i < result_atoms.size(); ++i)
            last.atoms[i].force = result_atoms[i].force;
          last.fweight = 1.0;
          const std::string none = "F: (-1 : -1)";
          const auto p = last.label.find(none);
          if (p != std::string::npos)
            last.label.replace(p, none.size(),
                               "F: (" + std::to_string(simLabel.forcesStartLine) +
                                   " : " + std::to_string(simLabel.forcesStopLine) + ")");
          last_has_forces = true;
          --stored_without_forces;
        }
        result_geometry_taken = false;
        clear_results();
      }
    }

    // Detect stress block
    if (line.find("Cartesian components (GPa)") != std::string::npos &&
        physical_stress) {
      if (debug) std::cout << "Reading stress..." << std::endl;
      take_result_geometry();
      if (!jumpNLines(stream, 3, "stress")) break;
      // A 3x3 matrix is expected
      simLabel.stressStartLine = line_counter + 1;
      for (int row = 0; row < 3; ++row) {
        if (!std::getline(stream, line)) {
          ends_inside("stress");
          break;
        }
        ++line_counter;
        std::istringstream iss(line);
        std::string ignore;
        double v1, v2, v3;
        if (!(iss >> ignore >> ignore >> v1 >> v2 >> v3)) {
          warn("the stress line " + std::to_string(row + 1) + ", " + quoted() +
               ", cannot be read; the structure is stored without stress");
          break;
        }
        s.stress(row, 0) = v1;
        s.stress(row, 1) = v2;
        s.stress(row, 2) = v3;
        simLabel.stressStopLine = line_counter;
        if (row == 2) stress_tensor_found = true;
      }
      // An MD step's stress printed after its MD Data ends the step's lines.
      if (md_step_ready) simLabel.stopLine = line_counter;
      if (current_state == CalcState::SinglePointEnergy ||
          (current_state == CalcState::FiniteBasisSetCorrection && !go_task) ||
          current_state == CalcState::BandStructure) {
        data_block_complete = true;
        simLabel.stopLine = line_counter;
      }
      // The final configuration's stress, printed after the last iteration
      // was stored (a fixed-cell run with 'stress calculation : on' prints
      // only this one): it belongs to that iteration when the geometry is
      // the same and the iteration has no stress of its own.
      if (current_state == CalcState::Idle && final_configuration && stored > 0) {
        if (stress_tensor_found && !last_has_stress && same_geometry_as_last()) {
          Structure &last = stdb(stdb.size() - 1);
          last.stress = s.stress;
          last.stress *= p_conv;  // GPa to eV/A^3
          last.sweight = 1.0;
          const std::string none = "S: (-1 : -1)";
          const auto p = last.label.find(none);
          if (p != std::string::npos)
            last.label.replace(p, none.size(),
                               "S: (" + std::to_string(simLabel.stressStartLine) +
                                   " : " + std::to_string(simLabel.stressStopLine) + ")");
          last_has_stress = true;
          --stored_without_stress;
        }
        final_configuration = false;
        result_geometry_taken = false;
        clear_results();
      }
    }

    // The final configuration of a geometry optimisation, and the end of an
    // MD run, whose forces and stress belong to the last stored step.
    if (current_state == CalcState::Idle &&
        (line.find(": Final Configuration") != std::string::npos ||
         (md_task && line.find("Finished MD") != std::string::npos))) {
      final_configuration = true;
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
      double v;
      if (!(iss >> ignore >> ignore >> ignore >> ignore >> ignore >> v))
        refuse("cannot read the energy from " + quoted());
      e_free = v;
      simLabel.energyLine = line_counter;
    }

    // The other energy lines of an SCF: 'Final energy, E = ...' (smeared),
    // 'Final energy = ...' (fixed occupancy, which prints no free energy:
    // with no smearing, E is F), and the dispersion-corrected 'Dispersion
    // corrected final energy* = ...' and 'Dispersion corrected final free
    // energy* (Ecor-TS) = ...' (DFT-D, SEDC).
    if (current_state == CalcState::GeometryOptimization
        || current_state == CalcState::SinglePointEnergy
        || current_state == CalcState::BandStructure
        || current_state == CalcState::FiniteBasisSetCorrection) {
      double *target = nullptr;
      if (line.rfind("Final energy, E", 0) == 0) {
        target = &e_smeared;
      } else if (line.rfind("Final energy", 0) == 0 &&
                 line.find_first_not_of(' ', 12) != std::string::npos &&
                 line[line.find_first_not_of(' ', 12)] == '=') {
        target = &e_fixed;
      } else if (line.rfind("Dispersion corrected final free energy*", 0) == 0) {
        target = &e_disp_free;
      } else if (line.rfind("Dispersion corrected final energy*", 0) == 0) {
        target = &e_disp;
      }
      if (target) {
        take_result_geometry();
        std::istringstream iss(line.substr(line.find('=') + 1));
        double v;
        if (line.find('=') == std::string::npos || !(iss >> v))
          refuse("cannot read the energy from " + quoted());
        *target = v;
        if (target != &e_smeared) simLabel.energyLine = line_counter;
      }
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
      if (!(iss >> ignore >> ignore >> ignore>> step >> ignore >> ignore >> ignore))
        refuse("cannot read the iteration number from " + quoted());
      simLabel.stepNumber = step_number(step);
      simLabel.stopLine = line_counter;
      data_block_complete = true;
    }

    // Detect potential energy line used in MD 
    if (line.find("Potential Energy:") != std::string::npos && current_state == CalcState::MolecularDynamics) {
      if (debug) std::cout << "Potential Energy:" << std::endl;
      take_result_geometry();
      std::istringstream iss(line);
      std::string ignore;
      double v;
      if (!(iss >> ignore >> ignore >> ignore >> v))
        refuse("cannot read the potential energy from " + quoted());
      e_md = v;
      simLabel.energyLine = line_counter;
    }

    // Detect temperature line in MD and finalize
    if (line.find("Temperature:") != std::string::npos && current_state == CalcState::MolecularDynamics) {
      if (debug) std::cout << "Temperature:" << std::endl;
      std::istringstream iss(line);
      std::string ignore;
      if (!(iss >> ignore >> ignore >> s.T))
        refuse("cannot read the temperature from " + quoted());
      simLabel.T = s.T;
      simLabel.stopLine = line_counter;
      md_step_ready = true;
    }

    // If a completed data set is detected, finalize
    if (data_block_complete) {
      finalize_structure();
      reset_flags(CalcState::Idle);
    }
  } // end while getline
  if (md_step_ready) finalize_structure();  // the file ends after an MD step
  log_missing_stress("CASTEP .castep " + simLabel.filename, stored_without_stress, stored);
  log_missing_forces("CASTEP .castep " + simLabel.filename, stored_without_forces, stored);
}

void CastepCastepReader::print_summary() const {
  std::cout << get_summary();
}

std::string CastepCastepReader::get_summary() const {
  return stdb.summary();
}
} // namespace mlip
} // namespace tadah
