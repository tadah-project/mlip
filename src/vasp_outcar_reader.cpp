#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/dataset_readers/missing_stress.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace tadah {
namespace mlip {

VaspOutcarReader::VaspOutcarReader(StructureDB& db) : DatasetReader(db) {}

VaspOutcarReader::VaspOutcarReader(StructureDB& db, const std::string& filename) 
: DatasetReader(db, filename), filename_(filename) {
  read_data(filename);
}

void VaspOutcarReader::read_data(const std::string& filename) {
  filename_ = filename;
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

void VaspOutcarReader::parse_data() {
  // order in OUTCAR is: VRHFIN, NIONS,...
  std::istringstream stream(raw_data_);
  std::string line;
  std::vector<std::string> atom_types;
  std::vector<size_t> natom_types;

  size_t natoms;;
  bool complete_structure = false;
  bool is_md = false;

  Structure s;
  // Ionic steps, counted where they end: an MD step at its EKIN_LAT line,
  // a relaxation step at its free-energy line. The label numbers them.
  size_t ionic_step = 0;
  bool have_energy = false;  // this step's first-principles free energy
  bool have_stress = false;  // this step's 'in kB' line
  size_t stored = 0, without_stress = 0;

  // Steps of VASP's machine-learned force field (ML_MODE = train/run) are
  // not first-principles data. Their blocks are marked 'ML' ('ML FORCE on
  // cell', 'TOTAL-FORCE (eV/Angst) (ML)', 'free  energy ML TOTEN'); they are
  // never read, and the steps are skipped with one warning at the end.
  bool ml_in_step = false;
  bool ml_stress_block = false;  // the next 'in kB' line is an ML one
  std::vector<size_t> skipped_ml_steps;

  // Starts the next step from scratch; the label so far (the file name,
  // set from the header for the first structure) is kept.
  auto discard_step = [&]() {
    const std::string label = s.label;
    s = Structure();
    s.label = label;
    have_energy = false;
    have_stress = false;
    ml_in_step = false;
  };

  while (std::getline(stream, line)) {
    if (line.find("molecular dynamics") != std::string::npos) {
      is_md = true;
      s.label += "MD ";
    }
    else if (line.find("VRHFIN") != std::string::npos) {
      std::string type = line.substr(line.find("=") + 1);
      type = type.substr(0, type.find(":"));
      atom_types.push_back(type);
      s.label += filename_ + " ";
    }

    else if (line.find("NIONS") != std::string::npos) {
      std::istringstream iss(line);
      std::string temp;
      int count = 0;

      // Process the line to reach the expected element
      while (iss >> temp && count < 11) {
        ++count;
        if (count == 11) {
          if (!(iss >> natoms)) {
            std::cerr << "Error: Failed to read number of atoms from line: " << line << std::endl;
          }
          break;
        }
      }

      // Check if the line was too short
      if (count < 11) {
        std::cerr << "Error: Line is too short to extract 'NIONS'. Line: " << line << std::endl;
      }
    }

    else if (line.find("ions per type = ") != std::string::npos) {
      std::istringstream iss(line);
      std::string temp;
      int count = 0;

      // Process the line to reach the expected element
      while (iss >> temp) {
        ++count;
        size_t ntype;
        if (count == 4) {
          for (size_t i=0; i<atom_types.size(); ++i) {
            iss >> ntype;
            natom_types.push_back(ntype);
          }
        }
      }

      if (atom_types.size() != natom_types.size()) {
        std::cerr << "Error:  \"ions per type\" and \"VRHFIN\" differ." << std::endl;
      }
      size_t total = std::accumulate(natom_types.begin(), natom_types.end(), static_cast<size_t>(0));
      if (total!=natoms) {
        std::cerr << "Error:  Sum of \"ions per type\" and \"NIONS\" differ." << std::endl;
      }

    }

    else if (line.find("FORCE on cell") != std::string::npos) {
      ml_stress_block = line.find("ML FORCE on cell") != std::string::npos;
      if (ml_stress_block) ml_in_step = true;
    }

    else if (line.find("in kB") != std::string::npos && ml_stress_block) {
      ml_stress_block = false;
    }

    else if (line.find("in kB") != std::string::npos) {
      std::vector<double> row(6); // xx yy zz xy yz zx
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5])) {
        // An unreadable tensor (e.g. VASP's '*****' overflow) is no stress.
        std::cerr << "Warning: VASP OUTCAR " << filename_
                  << ": the stress line '" << line
                  << "' cannot be read; the step is stored without stress"
                  << std::endl;
        s.stress.set_zero();
        have_stress = false;
      } else {
        have_stress = true;
        s.stress(0,0) = row[0], s.stress(1,1) = row[1], s.stress(2,2) = row[2];
        s.stress(0,1) = s.stress(1,0) = row[3]; // xy
        s.stress(1,2) = s.stress(2,1) = row[4]; // yz
        s.stress(0,2) = s.stress(2,0) = row[5]; // zx
        // VASP prints the pressure-signed tensor (positive under
        // compression); Tadah stores the Cauchy stress, positive under
        // tension: negate, and convert kB to eV/A^3.
        s.stress *= -s_conv;
      }
    }

    else if (line.find("direct lattice vectors") != std::string::npos) {
      // OUTCAR lists each lattice vector a_i on its own row; store as row i of cell.
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << i << std::endl;
          break;
        }
        std::istringstream iss(line);
        if (!(iss >> s.cell(i,0) >> s.cell(i,1) >> s.cell(i,2))) {
          std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << i << std::endl;
          break;
        }
      }
    }

    else if (line.find("TOTAL-FORCE") != std::string::npos &&
             line.find("ML") != std::string::npos) {
      ml_in_step = true;
    }

    else if (line.find("TOTAL-FORCE (eV/Angst)") != std::string::npos) {
      if (!std::getline(stream, line)) {
        std::cerr << "Warning: Unexpected end of data when reading atom information" << std::endl;
      }
      size_t count = 0;
      for (size_t j = 0; j < natom_types.size(); ++j) {
        tadah::core::Element element(atom_types[j]);
        for (size_t i = 0; i < natom_types[j]; ++i) {
          if (!std::getline(stream, line)) {
            std::cerr << "Warning: Unexpected end of data when reading atom information at atom " << count+1 << std::endl;
            break;
          }
          std::istringstream iss(line);
          double px,py,pz,fx,fy,fz;
          if (!(iss >> px >> py >> pz >> fx >> fy >> fz)) {
            std::cerr << "Warning: Unexpected end of data when reading atom information at atom " << count+1 << std::endl;
          } else {
            Atom atom(element, px, py, pz, fx, fy, fz);
            s.add_atom(atom);
            count++;
          }
        }
      }

      if (count != natoms) {
        std::cerr << "Warning: Unexpected end of data when reading atom information" << std::endl;
        std::cerr << "Warning: Total ATOMS: " << natoms << " but found " << count << std::endl;
      }

    }

    // The free energy F (TOTEN) of the ionic step, the energy whose
    // gradient the forces and the stress are. The ionic-step line has two
    // spaces between the words ('free  energy   TOTEN'), the per-SCF lines
    // one ('free energy    TOTEN'). A relaxation's structure is complete
    // here; an MD step's at its EKIN_LAT line.
    else if (line.find("free  energy   TOTEN  =") != std::string::npos) {
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> tmp >> tmp >> s.energy)) {
        std::cerr << "Warning: Unexpected end of data when reading the free energy" << std::endl;
      }
      else {
        have_energy = true;
        if (!is_md) {
          complete_structure = true;
          s.label += "Structure " + std::to_string(++ionic_step);
        }
      }
    }
    else if (line.find("free  energy ML TOTEN") != std::string::npos) {
      ml_in_step = true;
      if (!is_md) {  // a relaxation step of the force field ends here
        skipped_ml_steps.push_back(++ionic_step);
        discard_step();
      }
    }
    else if (line.find("EKIN_LAT=") != std::string::npos) {
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> tmp >> tmp >> tmp >> s.T)) {
        std::cerr << "Warning: Unexpected end of data when reading temperature" << std::endl;
      }
      else if (have_energy) {
        complete_structure = true;
        s.label += "Structure " + std::to_string(++ionic_step);
      }
      else if (ml_in_step) {
        skipped_ml_steps.push_back(++ionic_step);
        discard_step();
      }
      else {
        throw std::runtime_error(
            "VASP OUTCAR " + filename_ + ": ionic step " +
            std::to_string(ionic_step + 1) +
            " has no 'free  energy   TOTEN' line before its EKIN_LAT line "
            "(an incomplete step, or one this reader does not recognise)");
      }
    }

    if (complete_structure) {
      // A step without a stress tensor: zero stress and stress weight 0,
      // so that a STRESS fit ignores it.
      if (!have_stress) {
        s.stress.set_zero();
        s.sweight = 0.0;
        ++without_stress;
      }
      stdb.add(s);
      ++stored;
      complete_structure = false;
      s = Structure();
      have_energy = false;
      have_stress = false;
      ml_in_step = false;
    }

  }

  // An MD run cut off after a step's free energy, before its EKIN_LAT line.
  if (is_md && have_energy) {
    std::cerr << "WARNING: VASP OUTCAR " << filename_ << ": ionic step "
              << ionic_step + 1
              << ", the last, has no EKIN_LAT line (the file ends inside it) "
                 "and was skipped; "
              << stored << " structure(s) read." << std::endl;
  }
  log_missing_stress("VASP OUTCAR " + filename_, without_stress, stored);

  if (!skipped_ml_steps.empty()) {
    const size_t shown = std::min<size_t>(skipped_ml_steps.size(), 10);
    std::ostringstream steps;
    for (size_t i = 0; i < shown; ++i)
      steps << (i ? ", " : "") << skipped_ml_steps[i];
    if (shown < skipped_ml_steps.size()) steps << ", ...";
    const bool one = skipped_ml_steps.size() == 1;
    std::cerr << "WARNING: VASP OUTCAR " << filename_ << ": "
              << (one ? "ionic step " : "ionic steps ") << steps.str();
    if (!one) std::cerr << " (" << skipped_ml_steps.size() << " steps)";
    std::cerr << (one ? " was" : " were")
              << " computed by VASP's machine-learned force field (marked ML) "
                 "and skipped: only first-principles steps are stored."
              << std::endl;
  }
}

void VaspOutcarReader::print_summary() const {
  std::cout << get_summary();
}

std::string VaspOutcarReader::get_summary() const {
  return stdb.summary();
}
} // namespace mlip
} // namespace tadah
