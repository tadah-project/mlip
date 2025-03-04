#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/lammps_structure_writer.h>

LammpsStructureWriter::LammpsStructureWriter(StructureDB& db) : StructureWriter(db) {}

void LammpsStructureWriter::write_data(const std::string& filename, const size_t i) {

  if (i >= stdb.size()) {
    throw std::out_of_range("Index i is out of range.");
  }
    
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + filename);
  }

  const Structure &st = stdb(i);
 
  // compute number of atoms for a given element
  const auto &elements = st.get_unique_elements();
  std::map<std::string, size_t> nelements;
  std::map<std::string, size_t> type;
  // Initialize the count for each element
  for (const auto& element : elements) {
    nelements[element.symbol] = 0;
  }
  // then count...
  size_t t=0;
  for (const auto &atom : st) {
    nelements[atom.symbol]++; 
    if (type.find(atom.symbol) == type.end()) {
      type[atom.symbol] = ++t;
    }
  }

  // BEGIN OF LAMMPS HEADER
  file << st.label << std::endl;
  file << std::endl;
  file << st.natoms() << " atoms" << std::endl;
  file << nelements.size() << " atom types" << std::endl;
  file << std::endl;

  // General triclinic box format
  std::vector<std::string> abcvec = {"avec", "bvec", "cvec"};
  for (int i=0; i<3; ++i) {
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,0);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,1);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,2)
      << "    " << abcvec[i];
    file << std::endl;
  }
  file << "    0.0 0.0 0.0 abc origin" << std::endl;
  // END LAMMPS HEADER

  // BEGIN LAMMPS BODY
  file << std::endl;
  file << "Masses" << std::endl;
  file << std::endl;
  for (const auto &t: type) {
    file << t.second << " " << PeriodicTable::get_mass(t.first) << "    # " << t.first << std::endl;
  }

  file << std::endl;
  file << "Atoms # atomic" << std::endl;
  file << std::endl;
  // atomic: atom-ID [int, 1-Natoms]    atom-type [int, 1-Ntype]    x y z

  size_t idx=1;
  for (const auto &atom : st) {
    file << std::right << std::fixed << std::setw(w) << idx;
    file << std::right << std::fixed << std::setw(w) << type[atom.symbol];
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(0);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(1);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(2);
    file << std::endl;
    idx++;
  }
  // END LAMMPS BODY
  
  file.close();
}
