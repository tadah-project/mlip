#include <tadah/mlip/structure.h>
#include <tadah/core/periodic_table.h>

#include <stdexcept>
#include <cmath>

Structure::Structure() {
  PeriodicTable::initialize();
}

const Atom &Structure::operator()(const size_t i) const{
  return atoms[i];
}

void Structure::add_atom(Atom a) {
  atoms.push_back(a);
}
void Structure::remove_atom(const size_t i) {
  atoms.erase(atoms.begin()+i);
}

const Vec3d& Structure::nn_pos(const size_t i, const size_t n) const {
  return near_neigh_atoms[i][n].position;
}

size_t Structure::natoms() const {
  return atoms.size();
}

size_t Structure::nn_size(size_t i) const {
  return near_neigh_atoms[i].size();
}

int Structure::read(std::ifstream &ifs) {
  std::string line;
  // ignore extra empty lines
  std::getline(ifs,line);
  if(line.empty()) return 1;

  // OK, process structural data
  // read first line as a label
  label = line;

  // the second line could be energy or
  // a scalling factors eweight fweight sweight
  std::getline(ifs,line);
  std::stringstream stream(line);
  std::string temp;
  size_t count = 0;
  while(stream >> temp) { ++count;}

  if (count == 3) {
    stream.clear();
    stream.seekg(0, std::ios::beg);
    stream >> eweight >> fweight >> sweight;
    std::getline(ifs,line);
    std::stringstream stream2(line);
    // energy and T
    stream2 >> energy;
    if(stream2) stream2 >> T;
  }
  else if (count == 2) {
    stream.clear();
    stream.seekg(0, std::ios::beg);
    // energy and T
    stream >> energy >> T;
  }
  else {
    energy = std::stod(line);
  }

  // 3 lines, 9 elements are for the cell matrix
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      ifs >> cell(i,j);

  // 3 lines, 9 elements are for the stress matrix
  // if (use_stress)
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      ifs >> stress(i,j);

  // move to next line
  std::getline(ifs,line);

  // make sure atoms vector is empty
  atoms.clear();

  char symbol[3];
  double px,py,pz,fx,fy,fz;
  while(std::getline(ifs,line)) {
    if(line.empty()) break;
    //if(line == " ") break;
    if(line == "\r") break;     // detects windows newline

    std::istringstream tmp(line);
    tmp >> symbol >> px >> py >> pz >> fx >> fy >> fz;
    Element e = PeriodicTable::find_by_symbol(symbol);
    atoms.push_back(Atom(e,px,py,pz,fx,fy,fz));
    //Atom &a = atoms.back();

    //tmp >> a.label >> a.position(0) >> a.position(1) >> a.position(2);
    // if (use_force)
    //tmp >> a.force(0) >> a.force(1) >> a.force(2);
    //std::cout <<  a.force(0) << " " <<  a.force(1) << " " <<  a.force(2) << std::endl;;
  }
  return 0;
}

void Structure::read(const std::string fn) {
  std::ifstream ifs(fn);
  if (!ifs.is_open()) {
    throw std::runtime_error("File does not exist: "+fn);
  }
  read(ifs);
  ifs.close();
}

void Structure::save(std::ofstream &ofs) const {
  ofs << label << std::endl;
  ofs << eweight << " " << fweight << " " << sweight << std::endl;
  ofs << energy << std::endl;
  for (int i=0; i<3; ++i) {
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << cell(i,0);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << cell(i,1);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << cell(i,2);
    ofs << std::endl;
  }

  for (int i=0; i<3; ++i) {
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << stress(i,0);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << stress(i,1);
    ofs << std::fixed << std::right << std::setw(w) << std::setprecision(p) << stress(i,2);
    ofs << std::endl;
  }

  for (const Atom &a:atoms) {
    ofs << a.symbol[0] << a.symbol[1];
    ofs << std::setw(w-2) << std::fixed << std::right
      << std::setprecision(p) << a.position(0);
    ofs << std::setw(w) << std::fixed << std::right
      << std::setprecision(p) << a.position(1);
    ofs << std::setw(w) << std::fixed << std::right
      << std::setprecision(p) << a.position(2);
    ofs << std::setw(w) << std::fixed << std::right
      << std::setprecision(p) << a.force(0);
    ofs << std::setw(w) << std::fixed << std::right
      << std::setprecision(p) << a.force(1);
    ofs << std::setw(w) << std::fixed << std::right
      << std::setprecision(p) << a.force(1);
    ofs << std::endl;
  }
}

void Structure::save(const std::string fn) const {
  std::ofstream ofs(fn);
  save(ofs);
  ofs.close();
}
size_t Structure::get_nn_iindex(const size_t i, const size_t j, const size_t jj) const {
  Vec3d shift_ijj = -near_neigh_shift[i][jj];
  size_t ii = 0;

  for (size_t x=0; x<near_neigh_idx[j].size(); ++x) {
    if (near_neigh_idx[j][x] == i)
      if (near_neigh_shift[j][ii] == shift_ijj)
        break;
    ii++;
  }
  return ii;

}
double Structure::get_volume() const {
  return cell.row(0)*(cell.row(1).cross(cell.row(2)));
}
double Structure::get_density() const {
  double V = get_volume();
  V*=1e-24; // convert to cm^3
  double amu = 1.66053906660e-24; // g
  double mass = 0;
  for (const auto& a:atoms) mass += PeriodicTable::get_mass(a.Z);
  return amu*mass/V;
}

double Structure::get_temperature() const {
  return T;
}

double Structure::get_virial_pressure() const {
  return stress.trace()/get_volume()/3;
}

double Structure::get_pressure(double T, double kB) const {
  double vpress = get_virial_pressure();
  return vpress + natoms()*kB*T/get_volume();
}
bool Structure::operator==(const Structure& st) const
{
  double EPSILON = std::numeric_limits<double>::epsilon();
  bool result =
    cell.isApprox(st.cell, EPSILON)
    && stress.isApprox(st.stress, EPSILON)
    && natoms()==st.natoms()
    && (std::fabs(eweight-st.eweight) < EPSILON)
    && (std::fabs(fweight-st.fweight) < EPSILON)
    && (std::fabs(sweight-st.sweight) < EPSILON)
    && (std::fabs(energy-st.energy) < EPSILON)
    ;
  if (!result) return result;
  for (size_t i=0;i<natoms();++i) {
    result = atoms[i]==st.atoms[i];
    if (!result) return result;
  }
  return result;
}
bool Structure::is_the_same(const Structure& st, double thr) const
{
  bool result =
    cell.isApprox(st.cell, thr)
    && natoms()==st.natoms();
  if (!result) return result;

  size_t count=0;;
  for (size_t i=0;i<natoms();++i) {
    for (size_t j=i;j<natoms();++j) {
      result = atoms[i].is_the_same(st.atoms[j], thr);
      if (result) count++;
    }
  }

  return count==natoms() ? true : false;
}
int Structure::next_structure(std::ifstream &ifs) {
  std::string line;
  std::getline(ifs,line);
  if(line.empty()) return 0;

  std::getline(ifs,line);
  // the second line could be energy or
  // a scalling factors eweight fweight sweight
  std::stringstream stream(line);
  std::string temp;
  size_t count = 0;
  while(stream >> temp) { ++count;}
  // optional if second line is a weight
  if (count==3)
    std::getline(ifs,line);

  for (size_t i=0; i<6;++i)
    std::getline(ifs,line);

  int natoms=0;
  while(std::getline(ifs,line)) {
    if(line.empty()) break;
    if(line == "\r") break;     // detects windows newline
    natoms++;
  }
  return natoms;
}
void Structure::clear_nn() {
  near_neigh_atoms.clear();
  near_neigh_shift.clear();
  near_neigh_idx.clear();
}

std::vector<Atom>::iterator Structure::begin() { 
    return atoms.begin(); 
}

std::vector<Atom>::iterator Structure::end() { 
    return atoms.end(); 
}

std::vector<Atom>::const_iterator Structure::begin() const { 
    return atoms.cbegin(); 
}

std::vector<Atom>::const_iterator Structure::end() const { 
    return atoms.cend(); 
}
void Structure::dump_to_file(std::ostream& file, size_t prec) const {
  const int n = 5;
  file << label << std::endl;
  file << std::fixed << std::setprecision(prec);
  file << eweight << " " << fweight << " " << sweight << std::endl;
  file << energy << " " << T << std::endl;

  file
    << std::setw(prec+n) << cell(0,0) << " "
    << std::setw(prec+n) << cell(0,1) << " "
    << std::setw(prec+n) << cell(0,2) << " " << std::endl
    << std::setw(prec+n) << cell(1,0) << " "
    << std::setw(prec+n) << cell(1,1) << " "
    << std::setw(prec+n) << cell(1,2) << " " << std::endl
    << std::setw(prec+n) << cell(2,0) << " "
    << std::setw(prec+n) << cell(2,1) << " "
    << std::setw(prec+n) << cell(2,2) << " " << std::endl;

  file 
    << std::setw(prec+n) << stress(0,0) << " "
    << std::setw(prec+n) << stress(0,1) << " "
    << std::setw(prec+n) << stress(0,2) << " " << std::endl
    << std::setw(prec+n) << stress(1,0) << " "
    << std::setw(prec+n) << stress(1,1) << " "
    << std::setw(prec+n) << stress(1,2) << " " << std::endl
    << std::setw(prec+n) << stress(2,0) << " "
    << std::setw(prec+n) << stress(2,1) << " "
    << std::setw(prec+n) << stress(2,2) << " " << std::endl;

  for (const auto& a : atoms) {
    file << std::setw(2) << a.symbol << " "
      << std::setw(prec+n) << a.position[0] << " "
      << std::setw(prec+n) << a.position[1] << " "
      << std::setw(prec+n) << a.position[2] << " "
      << std::setw(prec+n) << a.force[0] << " "
      << std::setw(prec+n) << a.force[1] << " "
      << std::setw(prec+n) << a.force[2] << std::endl;
  }
  file << std::endl;
}
void Structure::dump_to_file(const std::string& filepath, size_t prec) const {
  std::ofstream file(filepath, std::ios::app);  // Open in append mode
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
    return;
  }
  dump_to_file(file,prec);
  file.close();
}
std::set<Element> Structure::get_unique_elements() const{
  std::set<Element> unique_elements;
  for (const auto& a:atoms) unique_elements.insert(a);
  return unique_elements;
}
