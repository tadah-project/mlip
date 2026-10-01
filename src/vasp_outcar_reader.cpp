#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
namespace tadah {
namespace mlip {

VaspOutcarReader::VaspOutcarReader(StructureDB& db) : DatasetReader(db) {}

VaspOutcarReader::VaspOutcarReader(StructureDB& db, const std::string& filename) 
: DatasetReader(db, filename), filename_(filename) {
  read_data(filename);
}

void VaspOutcarReader::read_data(const std::string& filename) {
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
  bool stress_tensor_bool = false;
  bool complete_structure = false;
  bool is_md = false;

  Structure s;
  size_t counter=0;

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

    else if (line.find("in kB") != std::string::npos) {
      stress_tensor_bool = true;
      std::vector<double> row(6); // xx yy zz xy yz zx
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5])) {
        std::cerr << "Warning: Unexpected end of data when reading stress tensor" << std::endl;
      } else {
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
      else if (!is_md) {
        complete_structure = true;
        s.label += "Structure " + std::to_string(++counter);
      }
    }
    else if (line.find("EKIN_LAT=") != std::string::npos) {
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> tmp >> tmp >> tmp >> s.T)) {
        std::cerr << "Warning: Unexpected end of data when reading temperature" << std::endl;
      }
      else {
        complete_structure = true;
        s.label += "Structure " + std::to_string(++counter);
      }
    }

    if (complete_structure) {
      stdb.add(s);
      complete_structure = false;
      s = Structure();
    }

  }

  if (!stress_tensor_bool) {
    for (auto &s : stdb) s.stress.set_zero();
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
