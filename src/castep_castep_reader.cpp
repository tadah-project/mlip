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
  // Enumerates possible parse states for line-by-line scanning
  enum class CalcState {
    Idle,               // No recognized computation
    GeometryOptimization,
    MolecularDynamics,
    SinglePoint         // SCF or single-point run
  };

  // Current parser state and relevant flags
  CalcState current_state = CalcState::Idle; 
  bool stress_tensor_found = false;
  bool data_block_complete = false; // Indicates a completed set of structure data
  bool is_stress = false;              // SCF run with stress tensor

  // Local counters and accumulators
  size_t natoms = 0;                // Number of atoms from the input
  int scf_counter = 0;              // Index for SCF runs (tracking distinct single-point steps)
  size_t line_counter = 0;          // Tracks line number for warnings
  bool debug = false;               // Local debug toggle

  // Temporary container to hold ongoing parse results
  Structure s;              
  std::vector<Atom> atoms;

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
    s.atoms.resize(atoms.size());
    for (size_t i = 0; i < atoms.size(); ++i) {
      s.atoms[i].position = s.cell * atoms[i].position;
      s.atoms[i].force = atoms[i].force;
      s.atoms[i].symbol = atoms[i].symbol;
    }
    stdb.add(s);
  };

  auto reset_flags = [&](CalcState new_state) {
    current_state = new_state;
    stress_tensor_found  = false;
    data_block_complete  = false;
    if (debug) std::cout << "Resetting flags for new calculation type" << std::endl;
  };

  // Use a string stream to iterate over lines in raw_data_ 
  std::istringstream stream(raw_data_);
  std::string line;
  while (std::getline(stream, line)) {
    ++line_counter;

    // Detect or store the "Unit Cell"
    if (line.find("Unit Cell") != std::string::npos) {
      // A new cell is encountered. Prepare to read the next three lines of vectors.
      // Finalize any incomplete data set if logic demands it, but not strictly necessary here
      if (debug) std::cout << "Reading 3 lines of Unit Cell" << std::endl;
      
      // Skip the next two lines, often blank or headers
      if (!std::getline(stream, line)) break; 
      ++line_counter;
      if (!std::getline(stream, line)) break; 
      ++line_counter;

      // Read the 3 lattice vectors
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Unexpected end of file while reading lattice vectors at row "
                    << i << " (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;
        std::istringstream iss(line);
        if (!(iss >> s.cell(0, i) >> s.cell(1, i) >> s.cell(2, i))) {
          std::cerr << "Warning: Failed to parse lattice vector at row "
                    << i << " (line: " << line_counter << ")." << std::endl;
          break;
        }
      }
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

    // Detect starting MD iteration
    if (line.find("Starting MD iteration") != std::string::npos) {
      reset_flags(CalcState::MolecularDynamics);
      std::istringstream iss(line);
      std::string ignore, step;
      iss >> ignore >> ignore >> ignore >> step;

      // Remove last step from label
      while (s.label.back() != ' ') 
        s.label.pop_back();

      s.label += step;

      if (debug) std::cout << "Starting MD iteration " << step << std::endl;
      continue;
    }

    // Detect starting SP calculation (zeroth step)
    if (line.find("single point energy") != std::string::npos) {
      reset_flags(CalcState::SinglePoint);
      if (!std::getline(stream, line)) break; 
      ++line_counter;
      std::istringstream iss(line);
      std::string ignore, stress;
      iss >> ignore >> ignore >> ignore >> stress;
      s.label = "File: " + filename + " | CASTEP single point calculation, stress: " + stress;
      if (stress == "on") {
        is_stress = true;
      }
      else {
        is_stress = false;
      }
      if (debug) std::cout << "single point energy" << std::endl;
    }

    // Detect starting GO iteration (zeroth step)
    if (line.find("Geometry Optimization Parameters") != std::string::npos) {
      reset_flags(CalcState::GeometryOptimization);
      std::string ignore, algo;

      if (!std::getline(stream, line)) break; 
      ++line_counter;
      if (!std::getline(stream, line)) break; 
      ++line_counter;
      std::istringstream iss(line);
      iss >> ignore >> ignore >> ignore >> algo;
      s.label = "File: " + filename + " | CASTEP GO, "+algo+ ", step: ";
      if (debug) std::cout << "Geometry Optimization Parameters" << std::endl;
    }

    // Detect starting MD simulation (zeroth step)
    if (line.find("Molecular Dynamics Parameters") != std::string::npos) {
      reset_flags(CalcState::MolecularDynamics);

      if (!std::getline(stream, line)) break; 
      ++line_counter;
      if (!std::getline(stream, line)) break; 
      ++line_counter;
      std::istringstream iss(line);
      std::string ignore, ensemble;
      iss >> ignore >> ignore >> ensemble;
      s.label = "File: " + filename + " | CASTEP MD, "+ensemble+" step: " + "0";
      if (debug) std::cout << "Molecular Dynamics Parameters" << std::endl;
      continue;
    }

    // Detect starting GO iteration
    if (line.find("Starting BFGS iteration") != std::string::npos || 
        line.find("Starting LBFGS iteration") != std::string::npos || 
        line.find("Starting Delocalized iteration") != std::string::npos || 
        line.find("Starting DampedMD iteration") != std::string::npos || 
        line.find("Starting TPSD iteration") != std::string::npos) {
      reset_flags(CalcState::GeometryOptimization);
      // This line often carries the iteration number
      std::istringstream iss(line);
      std::string ignore, algo, step_str;
      // The typical format: "Starting MD iteration N ..."
      // Attempt to parse out the iteration number as step_str
      iss >> ignore >> algo >> ignore >> step_str;
      // Store it in a label for clarity
      s.label = "File: " + filename + " | CASTEP GO, "+algo+" step: ";
      if (debug) std::cout << "Starting" +algo+ " iteration " << step_str << std::endl;
      continue;
    }

    // if (line.find("finished MD iteration") != std::string::npos) {
    //   reset_flags(CalcState::Idle);
    // }

    // Detect fractional coordinates block
    if (line.find("Fractional coordinates of atoms") != std::string::npos) {
      if (debug) std::cout << "Fractional coordinates of atoms" << std::endl;
      // Typically followed by two header/blank lines
      if (!std::getline(stream, line)) break; 
      ++line_counter;
      if (!std::getline(stream, line)) break; 
      ++line_counter;

      // If constant volume, the code does not always reprint the cell.  
      // The logic here re-uses the last known cell if needed. 
      atoms.clear(); // Clear old atoms or partial data

      atoms.reserve(natoms);
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
        Atom a(Element(species), px, py, pz, 0.0, 0.0, 0.0);
        atoms.push_back(a); // keep it fractional for now
      }
      continue;
    }

    // The old cell is reverted and printed before final enthalpy from the current iteration
    // We simply skip over this block of text
    if (line.find("reverting to earlier configuration") != std::string::npos) {
      if (debug) std::cout << "reverting to earlier configuration" << std::endl;
      for (int i = 0; i < 30; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Failed skiping over revered configuration " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;
      }
    }

    // Detect forces block
    if (line.find("Cartesian components (eV/A)") != std::string::npos) {
      if (debug) std::cout << "Reading forces..." << std::endl;
      // Usually there are three lines of header/blank text before actual force lines
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Incomplete forces header block (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;
      }
      // Read force for each atom
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
        // Format typically: " ??? ??? ??? fx fy fz "
        if (!(iss >> tmp >> tmp >> tmp >> fx >> fy >> fz)) {
          std::cerr << "Warning: Could not parse atomic forces (line: " 
                    << line_counter << ")." << std::endl;
          break;
        }
        // Assign force to the matching atom
        if (i < atoms.size()) {
          atoms[i].force = Vec3d(fx, fy, fz);
        }
      }
      continue;
    }

    // Detect stress block
    if (line.find("Cartesian components (GPa)") != std::string::npos) {
      if (debug) std::cout << "Reading stress..." << std::endl;
      // Usually there are three lines of header/blank text
      for (int i = 0; i < 3; ++i) {
        if (!std::getline(stream, line)) {
          std::cerr << "Warning: Incomplete stress header block (line: " << line_counter << ")." << std::endl;
          break;
        }
        ++line_counter;
      }
      // A 3x3 matrix is expected
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
        // Format: " ??? ??? val1 val2 val3 "
        if (!(iss >> ignore >> ignore >> v1 >> v2 >> v3)) {
          std::cerr << "Warning: Could not parse stress row (line: " 
                    << line_counter << ")." << std::endl;
          break;
        }
        s.stress(row, 0) = v1;
        s.stress(row, 1) = v2;
        s.stress(row, 2) = v3;
      }
      stress_tensor_found = true;
      if (current_state == CalcState::SinglePoint) {
        s.label += " SCF run: " + std::to_string(scf_counter++);
        if (is_stress) data_block_complete = true;
      }
      continue;
    }

    // Detect final free energy for single-point or last iteration in GO
    if (line.find("Final free energy (E-TS)") != std::string::npos && 
        (current_state == CalcState::GeometryOptimization || current_state == CalcState::SinglePoint)) {
      if (debug) std::cout << "Final free energy (E-TS)" << std::endl;
      // Single point or final iteration of a geometry optimization 
      std::istringstream iss(line);
      std::string ignore;
      double energy;
      // Typical: "Final free energy (E-TS) =    <val> eV"
      // Rolling through tokens to read last numeric:
      if (!(iss >> ignore >> ignore >> ignore >> ignore >> ignore >> energy)) {
        std::cerr << "Warning: Could not parse final free energy (line: "
                  << line_counter << ")." << std::endl;
      } else {
        s.energy = energy;
      }

      // Single-point detection: if current_state is Idle or SinglePoint, treat it as an SCF run
      // but it might also appear mid-GO. The code below sets a label if no other structure data is present.
      if (current_state == CalcState::SinglePoint && !is_stress) {
        s.label += " SCF run: " + std::to_string(scf_counter++);
        data_block_complete = true;
      }
      // Mark that data is complete for this structure
      continue;
    }

    // Detect the line with "enthalpy=" used in geometry optimization
    if (line.find("enthalpy=") != std::string::npos && current_state == CalcState::GeometryOptimization) {
      if (debug) std::cout << "enthalpy=" << std::endl;
      // This typically indicates a converged or final iteration of geometry optimization
      std::istringstream iss(line);
      std::string ignore, step;
      if (!(iss >> ignore >> ignore >> ignore>> step >> ignore >> ignore >> s.energy)) {
        std::cerr << "Warning: Could not parse final free energy (line: "
                  << line_counter << ")." << std::endl;
      }
      s.label += step;
      data_block_complete = true;
      continue;
    }

    // Detect potential energy line used in MD 
    if (line.find("Potential Energy:") != std::string::npos && current_state == CalcState::MolecularDynamics) {
      if (debug) std::cout << "Potential Energy:" << std::endl;
      // Format: "Potential Energy: <value>"
      std::istringstream iss(line);
      std::string ignore;
      double energy;
      if (!(iss >> ignore >> ignore >> ignore >> energy)) {
        std::cerr << "Warning: Could not parse potential energy in MD (line: " 
                  << line_counter << ")." << std::endl;
      } else {
        s.energy = energy;
      }
      continue;
    }

    // Detect temperature line in MD, which often appears last in an MD step
    if (line.find("Temperature:") != std::string::npos && current_state == CalcState::MolecularDynamics) {
      if (debug) std::cout << "Temperature:" << std::endl;
      // This usually concludes an MD iteration
      std::istringstream iss(line);
      std::string ignore;
      double temperature;
      // Format: "Temperature: <value>"
      if (!(iss >> ignore >> ignore >> temperature)) {
        std::cerr << "Warning: Could not parse temperature in MD (line: " 
                  << line_counter << ")." << std::endl;
      } else {
        s.T = temperature;
      }
      // If a label has not been set, create a fallback
      // if (s.label.empty()) {
      //   s.label = "CASTEP MD step (no step # found)";
      // }
      data_block_complete = true;
      continue;
    }

    // If a completed data set is detected, finalize
    if (data_block_complete) {
      finalize_structure();
      reset_flags(CalcState::Idle);
    }
  } // end while getline

  // At file end, if a data block was incomplete but has partial data, 
  // it can either be discarded or stored as partial. 
  // For safety, only store if data_block_complete was triggered.
  // No action is taken here unless a user-specific policy is desired.
}

// void CastepCastepReader::parse_data() {
//   std::istringstream stream(raw_data_);
//   std::string line;
//
//   size_t natoms;;
//   bool stress_tensor_bool = false;
//   bool constant_volume = false;
//   bool complete_structure = false;
//   bool md_run = false;
//   bool go_run = false;
//   bool restart = false;
//   int scf_idx = 0;
//
//   Structure s;
//   std::string label;
//   size_t counter=0;
//
//   bool debug = false;
//
//   while (std::getline(stream, line)) {
//
//     if (line.find("type of calculation") != std::string::npos) {
//       stress_tensor_bool=false;
//       constant_volume=false;
//       md_run = false;
//       go_run = false;
//       restart=false;
//       if (line.find(": geometry optimization") != std::string::npos) {
//         if (debug) std::cout << "GO run" << std::endl;
//         go_run = true;
//       }
//       if (line.find(": molecular dynamics") != std::string::npos) {
//         md_run = true;
//       }
//     }
//     else if (line.find("Restarting MD") != std::string::npos) {
//       restart=true;
//     }
//     else if (line.find("Unit Cell") != std::string::npos) {
//       if (debug) std::cout << "Unit Cell" << std::endl;
//       if (!std::getline(stream, line) || !std::getline(stream, line)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading atom information" << std::endl;
//       }
//       counter++;
//       counter++;
//
//       s = Structure();
//       for (int i = 0; i < 3; ++i) {
//         if (!std::getline(stream, line)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << i << std::endl;
//           break;
//         }
//         counter++;
//         std::istringstream iss(line);
//         if (!(iss >> s.cell(0,i) >> s.cell(1,i) >> s.cell(2,i))) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << i << std::endl;
//           break;
//         }
//       }
//
//       if (constant_volume) label.clear();
//       constant_volume = false;
//       if (debug) std::cout << s.cell << std::endl;
//     }
//
//     else if (line.find("Total number of ions in cell") != std::string::npos) {
//       // this line indicates new type of computation, so we reset flags here
//       std::istringstream iss(line);
//       std::string temp;
//       int count = 0;
//
//       // Process the line to reach the 8th element
//       while (iss >> temp) {
//         if (++count == 7 && !(iss >> natoms)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Error: Failed to read number of atoms from line: " << line << std::endl;
//         }
//       }
//
//       // Check if the line was too short
//       if (count < 7) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Error: Line is too short to extract total number of ions in a cell. Line: " << line << std::endl;
//       }
//       if (debug) std::cout << "Number of atoms: " << natoms << std::endl;
//     }
//
//     else if (line.find("so fixing cell parameters") != std::string::npos) {
//       if (debug) std::cout << "so fixing cell parameters" << std::endl;
//       constant_volume=true;
//       label = "CASTEP MD, const. volume: true, step: 0";
//     }
//
//     else if (line.find("Starting MD iteration") != std::string::npos) {
//       if (debug) std::cout << "Starting MD iteration" << std::endl;
//       std::istringstream iss(line);
//       std::string step;
//       iss >> step >> step >> step >> step;
//       std::ostringstream oss;
//       oss << std::boolalpha << constant_volume;
//       label = "CASTEP MD, const. volume: " + oss.str() + ", step: " + step;
//     }
//
//
//     else if (constant_volume && line.find("Cell Contents") != std::string::npos) {
//       if (debug) std::cout << "Cell Contents" << std::endl;
//       if (restart) {
//         if (debug) std::cout << "Cell Contents restart" << std::endl;
//         s.atoms.clear();
//       } else {
//         if (debug) std::cout << "Cell Contents no restart" << std::endl;
//         if (debug) std::cout << line << std::endl;
//         s = Structure();
//         s.cell = stdb.structures.back().cell;   // copy last cell as it is not repeated in castep for const. volume
//       }
//     }
//
//     else if (line.find("Fractional coordinates of atoms") != std::string::npos) {
//       if (debug) std::cout << "Fractional coordinates of atoms" << std::endl;
//       // Ideally st.natoms() should be zero by now. However for constant volum GO this is not the case.
//       // Therfore we need to clear the atoms vector but we keep the st.cell as it is.
//       s.atoms.clear();
//       if (!std::getline(stream, line) || !std::getline(stream, line)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading atom information" << std::endl;
//       }
//       counter++;
//       counter++;
//
//       for (size_t i = 0; i < natoms; ++i) {
//         if (!std::getline(stream, line)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading atomic coordinates at row " << i << std::endl;
//           break;
//         }
//         counter++;
//         std::istringstream iss(line);
//         std::string type, tmp;
//         double px,py,pz;
//         if (!(iss >> tmp >> type >> tmp >> px >> py >> pz)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading atomic coordiantes at row " << i << std::endl;
//           break;
//         }
//         s.add_atom(Atom(Element(type),px,py,pz,0,0,0));
//         s.atoms[i].position = s.cell * s.atoms[i].position;  // convert to abs
//       }
//       if (debug) std::cout << "natoms: " << s.natoms() << std::endl;
//     }
//
//     else if (line.find("Cartesian components (eV/A)") != std::string::npos) {
//       if (debug) std::cout << "Cartesian components (eV/A)" << std::endl;
//       if (s.natoms()!=natoms) continue;
//       if (!std::getline(stream, line) || !std::getline(stream, line) || !std::getline(stream, line)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading atom forces" << std::endl;
//       }
//       counter++;
//       counter++;
//       counter++;
//       for (size_t i = 0; i < natoms; ++i) {
//         if (!std::getline(stream, line)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading atom forces: " << line << std::endl;
//           break;
//         }
//         counter++;
//         std::istringstream iss(line);
//         std::string tmp;
//         double fx,fy,fz;
//         if (!(iss >> tmp >> tmp >> tmp >> fx >> fy >> fz)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading atom forces: " << line << std::endl;
//         } else {
//           s.atoms[i].force = Vec3d(fx,fy,fz);
//         }
//       }
//     }
//
//     else if (line.find("Cartesian components (GPa)") != std::string::npos) {
//       if (debug) std::cout << "Cartesian components (GPa)" << std::endl;
//       if (!std::getline(stream, line) || !std::getline(stream, line) || !std::getline(stream, line)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading stress tensor" << std::endl;
//       }
//       counter++;
//       counter++;
//       counter++;
//       for (size_t i = 0; i < 3; ++i) {
//         if (!std::getline(stream, line)) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading stress tensor: " << line << std::endl;
//           break;
//         }
//         counter++;
//         std::istringstream iss(line);
//         std::string tmp;
//         if (!(iss >> tmp >> tmp >> s.stress(i,0) >> s.stress(i,1) >> s.stress(i,2))) {
//           std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//           std::cerr << "Warning: Unexpected end of data when reading atom forces: " << line << std::endl;
//         } 
//       }
//       stress_tensor_bool = true;
//     }
//     // Potential Energy // MD
//     // Total energy corrected for finite basis set =   -2690.182502 eV  // GO, first step only
//     // enthalpy=  indicates end of iteration for GO
//
//     else if (line.find("Final free energy (E-TS)") != std::string::npos) {
//       if (debug) std::cout << "Final free energy (E-TS)" << std::endl;
//       std::istringstream iss(line);
//       std::string tmp;
//       if (!(iss >> tmp >> tmp >> tmp >> tmp >> tmp >> s.energy)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading total energy" << std::endl;
//       }
//       // scf only loop
//       if (!md_run && !go_run) {
//         s.label = "SCF run: " + std::to_string(scf_idx++);
//         complete_structure = true;
//       }
//     }
//
//     else if (line.find("enthalpy=") != std::string::npos) {
//       if (debug) std::cout << "enthalpy=" << std::endl;
//       // GO: the last extracted free energy is correct one!
//       std::istringstream iss(line);
//       std::string step;
//       iss >> step >> step >> step >> step;  // safe as found enthalpy=
//       std::ostringstream oss;
//       oss << std::boolalpha << constant_volume;
//       s.label = "CASTEP GO, const. volume: " + oss.str() + ", step: " + step;
//       complete_structure = true;
//     }
//
//     else if (line.find("Potential Energy:") != std::string::npos) {
//       if (debug) std::cout << "Potential Energy:" << std::endl;
//       std::istringstream iss(line);
//       std::string tmp;
//       if (!(iss >> tmp >> tmp >> tmp >> s.energy)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading total energy" << std::endl;
//       }
//     }
//     else if (line.find("Temperature:") != std::string::npos) {
//       if (debug) std::cout << "Temperature:" << std::endl;
//       // MD: end of iteration
//       std::istringstream iss(line);
//       std::string tmp;
//       if (!(iss >> tmp >> tmp >> s.T)) {
//         std::cerr << "Warning, file" << filename << " line: " << counter << std::endl;
//         std::cerr << "Warning: Unexpected end of data when reading temperature" << std::endl;
//       }
//
//       if (!label.size())label = "CASTEP MD, const. volume: false, step: 0"; // the last option
//       s.label = label;
//       complete_structure = true;
//     }
//     if (complete_structure) {
//       if (!stress_tensor_bool) {
//         s.stress.set_zero();
//       }
//       else  {
//         s.stress *= p_conv; // GPa -> eV/A^3
//       }
//       stdb.add(s);
//       complete_structure = false;
//     }
//     counter++;
//   }
//
// }

void CastepCastepReader::print_summary() const {
  std::cout << get_summary();
}

std::string CastepCastepReader::get_summary() const {
  return stdb.summary();
}
