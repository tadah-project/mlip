#include <tadah/core/utils.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>

CastepMDReader::CastepMDReader(StructureDB& db)
: DatasetReader(db) {}

CastepMDReader::CastepMDReader(StructureDB& db, const std::string& filename)
: DatasetReader(db, filename) {
  read_data(filename);
  parse_data();
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
  // T /= k_b;

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
  return "Time: " + std::to_string(std::stod(time)*t_conv) + " ps";
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
      s.label = get_first_label(time);
      iss >> s.energy;
      s.energy *= e_conv;
      break;
    }
    iss >> time;
  }

  while (std::getline(stream, line)) {
    if (ends_with(line,"<-- T")) {
      std::istringstream iss(line);
      iss >> T;
    }
    else if (ends_with(line,"<-- E")) {
      std::istringstream iss(line);
      iss >> s.energy;
      s.energy *= e_conv;
    }
    else if (ends_with(line,"<-- h")) {
      //  the current matrix of cell vectors 
      std::istringstream iss(line);
      if (!(iss >> s.cell(0,cell_idx) >> s.cell(1,cell_idx) >> s.cell(2,cell_idx)))
        std::cerr << "Warning: Unexpected end of data when reading lattice vectors at row " << cell_idx << std::endl;
      cell_idx++;
    }

    else if (ends_with(line,"<-- S")) {
      // The full pressure tensor (including kinetic contributions)
      std::istringstream iss(line);
      if (!(iss >> s.stress(0,stress_idx) >> s.stress(1,stress_idx) >> s.stress(2,stress_idx)))
        std::cerr << "Warning: Unexpected end of data when reading stress tensor at row " << stress_idx << std::endl;
      stress_idx++;
      stress_tensor_bool = true;
    }
    else if (ends_with(line,"<-- R")) {
      // First the position vectors of all ions are printed with the label <-- R.
      std::string element;
      int temp;
      double px, py, pz;
      std::istringstream iss(line);
      if (!(iss >> element >> temp >> px >> py >> pz)) 
        std::cerr << "Warning: Unexpected end of data when reading atomic positions:\n" << line << std::endl;
      s.add_atom(Atom(Element(element),px*d_conv,py*d_conv,pz*d_conv,0,0,0));
    }
    else if (ends_with(line,"<-- F")) {
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
    }
    else if (is_blank_line(line)) {
      postproc_structure(s);

      if (std::getline(stream, line)) {
        std::string time;
        std::istringstream iss(line);
        iss >> time;
        s.label = get_label(time);
      }

      cell_idx=0;
      stress_idx=0;
      force_idx=0;

    }
  }

  // in case there is no blank line at the end we have to add last strcuture here
  if (s.natoms()) {
    postproc_structure(s);
  }
}

void CastepMDReader::print_summary() const {
  std::cout << stdb;
}
