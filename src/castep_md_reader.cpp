#include <tadah/core/utils.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>

CastepMDReader::CastepMDReader(StructureDB& db)
: DatasetReader(db) {}

CastepMDReader::CastepMDReader(StructureDB& db, const std::string& filename)
: DatasetReader(db, filename) {
  read_data(filename);
}

void CastepMDReader::read_data(const std::string& filename) {
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

bool CastepMDReader::ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() && 
  str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

double CastepMDReader::calc_P_ideal(Structure &s, double T) {
  double V = s.get_volume();  // Bohr^3
  int N = s.natoms();
  return N*k_b*T/V;
}
void CastepMDReader::postproc_structure(Structure &s) {
  if (!stress_tensor_bool) {
    s.stress.set_zero();
  } else {
    // stress in .md file contains kinetic contributions
    // so we remove it to obtain pure virial stress tensor
    // P = N k_b T / V
    s.stress -= calc_P_ideal(s,T);
  }

  // finish conversion
  s.cell *= d_conv;
  s.stress *= s_conv;
  s.T = T/k_b;

  // add to database
  stdb.add(s);

  // reset
  stress_tensor_bool = false;
  s = Structure();
}

std::string CastepMDReader::get_first_label(std::string &time) {
  return "Filename: "+filename_+ " | Units: (eV, Angstrom) | Time: "
  + std::to_string(std::stod(time)*t_conv) + " ps";
}
std::string CastepMDReader::get_label(std::string &time) {
  try {
    return "Time: " + std::to_string(std::stod(time)*t_conv) + " ps";
  } catch (const std::invalid_argument& e) {
    std::cerr << "Warning: " << filename << std::endl;
    std::cerr << "Invalid argument: unable to convert to double: " << time << std::endl;
  } catch (const std::out_of_range& e) {
    std::cerr << "Warning: " << filename << std::endl;
    std::cerr << "Out of range: string represents a value too large: " << time << std::endl;
  }
  return "Time: FAIL" ;
}

void CastepMDReader::parse_data() {
  std::istringstream stream(raw_data_);
  std::string line;

  Structure s;
  size_t cell_idx=0;
  size_t stress_idx=0;
  size_t force_idx=0;

  // skip header if exists also deal with the case 
  // where there is no blank line at the begining of the .md file
  std::string time;
  while (std::getline(stream, line)) {
    std::istringstream iss(line);
    if (ends_with(line,"<-- E")) {
      iss >> s.energy;
      s.energy *= e_conv;
      break;
    }
    iss >> time;
  }
  bool error=false;
  //bool T_flag=false;
  bool E_flag=false;
  int H_flag=0;
  int S_flag=0;
  bool R_flag=false;
  bool F_flag=false;
  bool T_flag=false;
  bool complete_structure=false;

  while (std::getline(stream, line)) {
    if (ends_with(line,"<-- T")) {
      if (T_flag || complete_structure) {
        error=true;
        continue;
      }
      std::istringstream iss(line);
      iss >> T;
      T_flag=true;
    }
    else if (ends_with(line,"<-- E")) {
      if (/* T_flag || */ E_flag || complete_structure) {
        error=true;
        continue;
      }
      std::istringstream iss(line);
      iss >> s.energy;
      s.energy *= e_conv;
      E_flag=true;
    }
    else if (ends_with(line,"<-- h")) {
      //  the current matrix of cell vectors 
      if (/* !T_flag || */ !E_flag || H_flag==3 || complete_structure) {
        error=true;
        continue;
      }
      std::istringstream iss(line);
      if (!(iss >> s.cell(0,cell_idx) >> s.cell(1,cell_idx) >> s.cell(2,cell_idx)))
        std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << cell_idx << std::endl;
      cell_idx++;
      H_flag++;
    }

    else if (ends_with(line,"<-- S")) {
      // The full pressure tensor (including kinetic contributions)
      if (/* !T_flag || */ !E_flag || H_flag!=3 || S_flag==3 || complete_structure) {
        error=true;
        continue;
      }
      std::istringstream iss(line);
      if (!(iss >> s.stress(0,stress_idx) >> s.stress(1,stress_idx) >> s.stress(2,stress_idx)))
        std::cerr << "Warning: Unexpected end of data when reading stress tensor at row " << stress_idx << std::endl;
      stress_idx++;
      stress_tensor_bool = true;
      S_flag++;
    }
    else if (ends_with(line,"<-- R")) {
      if (/* !T_flag || */ !E_flag || H_flag!=3 || F_flag || complete_structure) {
        error=true;
        continue;
      }
      // First the position vectors of all ions are printed with the label <-- R.
      std::string element;
      int temp;
      double px, py, pz;
      std::istringstream iss(line);
      if (!(iss >> element >> temp >> px >> py >> pz)) 
        std::cerr << "Warning: Unexpected end of data when reading atomic positions:\n" << line << std::endl;
      s.add_atom(Atom(Element(element),px*d_conv,py*d_conv,pz*d_conv,0,0,0));
      R_flag=true;
    }
    else if (ends_with(line,"<-- F")) {
      if (/* !T_flag || */ !E_flag || H_flag!=3 || !R_flag || force_idx==s.natoms() || complete_structure) {
        error=true;
        continue;
      }
      std::string element;
      int temp;
      double fx, fy, fz;
      std::istringstream iss(line);
      if (!(iss >> element >> temp >> fx >> fy >> fz)) 
        std::cerr << "Warning: Unexpected end of data when reading atomic forces:\n" << line << std::endl;
      s.atoms[force_idx].force[0]=fx*f_conv;
      s.atoms[force_idx].force[1]=fy*f_conv;
      s.atoms[force_idx].force[2]=fz*f_conv;
      force_idx++;
      F_flag=true;
      if (force_idx==s.natoms()) complete_structure=true;
    }
    else if (is_blank_line(line)) {
      if (!error && complete_structure)
        postproc_structure(s);

      error=false;
      T_flag=false;
      E_flag=false;
      H_flag=0;
      S_flag=0;
      R_flag=false;
      F_flag=false;
      complete_structure=false;


      if (std::getline(stream, line)) {
        std::string time;
        std::istringstream iss(line);
        iss >> time;
        s.label = !stdb.size() ? get_first_label(time) : get_label(time);
      }

      cell_idx=0;
      stress_idx=0;
      force_idx=0;

    }
  }

  // in case there is no blank line at the end we have to add last strcuture here
  if (s.natoms() && !error && complete_structure) {
    postproc_structure(s);
  }
}

void CastepMDReader::print_summary() const {
  std::cout << get_summary();
}

std::string CastepMDReader::get_summary() const {
  return stdb.summary();
}
