#include "tadah/core/element.h"
#include "tadah/core/utils.h"
#include <string>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>

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
  std::istringstream stream(raw_data_);
  std::string line;

  size_t natoms;;
  bool stress_tensor_bool = false;
  bool constant_volume = false;
  bool complete_structure = false;
  bool md_run = false;
  bool go_run = false;
  bool restart = false;
  int scf_idx = 0;

  Structure s;
  std::string label;
  size_t counter=0;

  bool debug = false;

  while (std::getline(stream, line)) {

    if (line.find("type of calculation") != std::string::npos) {
      stress_tensor_bool=false;
      constant_volume=false;
      md_run = false;
      go_run = false;
      restart=false;
      if (line.find(": geometry optimization") != std::string::npos) {
        if (debug) std::cout << "GO run" << std::endl;
        go_run = true;
      }
      if (line.find(": molecular dynamics") != std::string::npos) {
        md_run = true;
      }
    }
    else if (line.find("Restarting MD") != std::string::npos) {
      restart=true;
    }
    else if (line.find("Unit Cell") != std::string::npos) {
      if (debug) std::cout << "Unit Cell" << std::endl;
      if (!std::getline(stream, line) || !std::getline(stream, line)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading atom information" << std::endl;
      }
      counter++;
      counter++;

      s = Structure();
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << i << std::endl;
          break;
        }
        counter++;
        std::istringstream iss(line);
        if (!(iss >> s.cell(0,i) >> s.cell(1,i) >> s.cell(2,i))) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << i << std::endl;
          break;
        }
      }

      if (constant_volume) label.clear();
      constant_volume = false;
      if (debug) std::cout << s.cell << std::endl;
    }

    else if (line.find("Total number of ions in cell") != std::string::npos) {
      // this line indicates new type of computation, so we reset flags here
      std::istringstream iss(line);
      std::string temp;
      int count = 0;

      // Process the line to reach the 8th element
      while (iss >> temp) {
        if (++count == 7 && !(iss >> natoms)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Error: Failed to read number of atoms from line: " << line << std::endl;
        }
      }

      // Check if the line was too short
      if (count < 7) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Error: Line is too short to extract total number of ions in a cell. Line: " << line << std::endl;
      }
      if (debug) std::cout << "Number of atoms: " << natoms << std::endl;
    }

    else if (line.find("so fixing cell parameters") != std::string::npos) {
      if (debug) std::cout << "so fixing cell parameters" << std::endl;
      constant_volume=true;
      label = "CASTEP MD, const. volume: true, step: 0";
    }

    else if (line.find("Starting MD iteration") != std::string::npos) {
      if (debug) std::cout << "Starting MD iteration" << std::endl;
      std::istringstream iss(line);
      std::string step;
      iss >> step >> step >> step >> step;
      std::ostringstream oss;
      oss << std::boolalpha << constant_volume;
      label = "CASTEP MD, const. volume: " + oss.str() + ", step: " + step;
    }


    else if (constant_volume && line.find("Cell Contents") != std::string::npos) {
      if (debug) std::cout << "Cell Contents" << std::endl;
      if (restart) {
        if (debug) std::cout << "Cell Contents restart" << std::endl;
        s.atoms.clear();
      } else {
        if (debug) std::cout << "Cell Contents no restart" << std::endl;
        if (debug) std::cout << line << std::endl;
        s = Structure();
        s.cell = stdb.structures.back().cell;   // copy last cell as it is not repeated in castep for const. volume
      }
    }

    else if (line.find("Fractional coordinates of atoms") != std::string::npos) {
      if (debug) std::cout << "Fractional coordinates of atoms" << std::endl;
      // Ideally st.natoms() should be zero by now. However for constant volum GO this is not the case.
      // Therfore we need to clear the atoms vector but we keep the st.cell as it is.
      s.atoms.clear();
      if (!std::getline(stream, line) || !std::getline(stream, line)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading atom information" << std::endl;
      }
      counter++;
      counter++;

      for (size_t i = 0; i < natoms; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading atomic coordinates at row " << i << std::endl;
          break;
        }
        counter++;
        std::istringstream iss(line);
        std::string type, tmp;
        double px,py,pz;
        if (!(iss >> tmp >> type >> tmp >> px >> py >> pz)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading atomic coordiantes at row " << i << std::endl;
          break;
        }
        s.add_atom(Atom(Element(type),px,py,pz,0,0,0));
        s.atoms[i].position = s.cell * s.atoms[i].position;  // convert to abs
      }
      if (debug) std::cout << "natoms: " << s.natoms() << std::endl;
    }

    else if (line.find("Cartesian components (eV/A)") != std::string::npos) {
      if (debug) std::cout << "Cartesian components (eV/A)" << std::endl;
      if (s.natoms()!=natoms) continue;
      if (!std::getline(stream, line) || !std::getline(stream, line) || !std::getline(stream, line)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading atom forces" << std::endl;
      }
      counter++;
      counter++;
      counter++;
      for (size_t i = 0; i < natoms; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading atom forces: " << line << std::endl;
          break;
        }
        counter++;
        std::istringstream iss(line);
        std::string tmp;
        double fx,fy,fz;
        if (!(iss >> tmp >> tmp >> tmp >> fx >> fy >> fz)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading atom forces: " << line << std::endl;
        } else {
          s.atoms[i].force = Vec3d(fx,fy,fz);
        }
      }
    }

    else if (line.find("Cartesian components (GPa)") != std::string::npos) {
      if (debug) std::cout << "Cartesian components (GPa)" << std::endl;
      if (!std::getline(stream, line) || !std::getline(stream, line) || !std::getline(stream, line)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading stress tensor" << std::endl;
      }
      counter++;
      counter++;
      counter++;
      for (size_t i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading stress tensor: " << line << std::endl;
          break;
        }
        counter++;
        std::istringstream iss(line);
        std::string tmp;
        if (!(iss >> tmp >> tmp >> s.stress(i,0) >> s.stress(i,1) >> s.stress(i,2))) {
          std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
          std::cerr << "Warning: Unexpected end of data when reading atom forces: " << line << std::endl;
        } 
      }
      stress_tensor_bool = true;
    }
    // Potential Energy // MD
    // Total energy corrected for finite basis set =   -2690.182502 eV  // GO, first step only
    // enthalpy=  indicates end of iteration for GO

    else if (line.find("Final free energy (E-TS)") != std::string::npos) {
      if (debug) std::cout << "Final free energy (E-TS)" << std::endl;
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> tmp >> tmp >> tmp >> s.energy)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading total energy" << std::endl;
      }
      // scf only loop
      if (!md_run && !go_run) {
        s.label = "SCF run: " + std::to_string(scf_idx++);
        complete_structure = true;
      }
    }

    else if (line.find("enthalpy=") != std::string::npos) {
      if (debug) std::cout << "enthalpy=" << std::endl;
      // GO: the last extracted free energy is correct one!
      std::istringstream iss(line);
      std::string step;
      iss >> step >> step >> step >> step;  // safe as found enthalpy=
      std::ostringstream oss;
      oss << std::boolalpha << constant_volume;
      s.label = "CASTEP GO, const. volume: " + oss.str() + ", step: " + step;
      complete_structure = true;
    }

    else if (line.find("Potential Energy:") != std::string::npos) {
      if (debug) std::cout << "Potential Energy:" << std::endl;
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> tmp >> s.energy)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading total energy" << std::endl;
      }
    }
    else if (line.find("Temperature:") != std::string::npos) {
      if (debug) std::cout << "Temperature:" << std::endl;
      // MD: end of iteration
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> s.T)) {
        std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
        std::cerr << "Warning: Unexpected end of data when reading temperature" << std::endl;
      }

      if (!label.size())label = "CASTEP MD, const. volume: false, step: 0"; // the last option
      s.label = label;
      complete_structure = true;
    }
    if (complete_structure) {
      if (!stress_tensor_bool) {
        s.stress.set_zero();
      }
      else  {
        s.stress *= p_conv; // GPa -> eV/A^3
      }
      stdb.add(s);
      complete_structure = false;
    }
    counter++;
  }

}

void CastepCastepReader::print_summary() const {
  std::cout << get_summary();
}

std::string CastepCastepReader::get_summary() const {
  return stdb.summary();
}
