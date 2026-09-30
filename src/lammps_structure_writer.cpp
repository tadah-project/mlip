#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/lammps_structure_writer.h>

#include <cmath>
namespace tadah {
namespace mlip {

LammpsStructureWriter::LammpsStructureWriter(StructureDB& db) : StructureWriter(db) {}

void LammpsStructureWriter::write_data(const std::string& filename, const size_t i) {

  if (i >= stdb.size()) {
    throw std::out_of_range("Index i is out of range.");
  }

  const Structure &st = stdb(i);

  // The rows of the cell are the lattice vectors. A LAMMPS general
  // triclinic box needs a right-handed, non-degenerate (avec, bvec, cvec):
  // a left-handed cell is written as (a0, a1, -a2) from origin a2, the
  // same parallelepiped (so every atom stays inside the box and each
  // lattice direction keeps its boundary flag), and a singular cell is
  // refused.
  const core::Vec3d a0 = st.cell.row(0), a1 = st.cell.row(1),
                    a2 = st.cell.row(2);
  const double det = a0 * a1.cross(a2);
  if (!(std::fabs(det) > 1e-12 * a0.norm() * a1.norm() * a2.norm())) {
    throw std::runtime_error("LAMMPS data export: structure '" + st.label +
                             "' has a singular cell (coplanar or zero "
                             "lattice vectors).");
  }
  const double csign = det < 0.0 ? -1.0 : 1.0;

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + filename);
  }
 
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

  // General triclinic box format. Reals are written with NUMERIC
  // significant digits (read_data parses E-notation; 18 is exact).
  std::vector<std::string> abcvec = {"avec", "bvec", "cvec"};
  for (int i=0; i<3; ++i) {
    const double f = (i == 2) ? csign : 1.0;
    put_sig(file, f * st.cell(i,0));
    put_sig(file, f * st.cell(i,1));
    put_sig(file, f * st.cell(i,2));
    file << "    " << abcvec[i];
    file << std::endl;
  }
  if (csign < 0.0) {
    put_sig(file, st.cell(2,0));
    put_sig(file, st.cell(2,1));
    put_sig(file, st.cell(2,2));
    file << "    abc origin" << std::endl;
  } else {
    file << "    0.0 0.0 0.0 abc origin" << std::endl;
  }
  // END LAMMPS HEADER

  // BEGIN LAMMPS BODY
  file << std::endl;
  file << "Masses" << std::endl;
  file << std::endl;
  for (const auto &t: type) {
    file << t.second << " ";
    put_sig(file, tadah::core::PeriodicTable::get_mass(t.first), false);
    file << "    # " << t.first << std::endl;
  }

  file << std::endl;
  file << "Atoms # atomic" << std::endl;
  file << std::endl;
  // atomic: atom-ID [int, 1-Natoms]    atom-type [int, 1-Ntype]    x y z

  size_t idx=1;
  for (const auto &atom : st) {
    file << std::right << std::setw(w) << idx;
    file << std::right << std::setw(w) << type[atom.symbol];
    put_sig(file, atom.position(0));
    put_sig(file, atom.position(1));
    put_sig(file, atom.position(2));
    file << std::endl;
    idx++;
  }
  // END LAMMPS BODY
  
  file.close();
}
} // namespace mlip
} // namespace tadah
