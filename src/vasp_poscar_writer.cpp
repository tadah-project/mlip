#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/vasp_poscar_writer.h>

#include <cmath>
#include <fstream>
namespace tadah {
namespace mlip {

VaspPoscarWriter::VaspPoscarWriter(StructureDB& db) : StructureWriter(db) {}

void VaspPoscarWriter::write_data(const std::string& filename, const size_t i) {

  if (i >= stdb.size()) {
    throw std::out_of_range("Index i is out of range.");
  }

  const Structure &st = stdb(i);

  // The rows of the cell are the lattice vectors. VASP stops on a
  // left-handed basis (negative triple product), so such a cell is written
  // as (a0, a1, -a2), the same lattice right-handed; the Cartesian
  // positions are unchanged. A singular cell is refused.
  const core::Vec3d a0 = st.cell.row(0), a1 = st.cell.row(1),
                    a2 = st.cell.row(2);
  const double det = a0 * a1.cross(a2);
  if (!(std::fabs(det) > 1e-12 * a0.norm() * a1.norm() * a2.norm())) {
    throw std::runtime_error("VASP POSCAR export: structure '" + st.label +
                             "' has a singular cell (coplanar or zero "
                             "lattice vectors).");
  }
  const double csign = det < 0.0 ? -1.0 : 1.0;

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + filename);
  }

  // write scaling factor
  file << st.label << std::endl;
  file << "1.0" << std::endl;

  // write cell. Reals are written with NUMERIC significant digits (POSCAR
  // is read list-directed: Fortran accepts E-notation; 18 is exact).
  for (int i=0; i<3; ++i) {
    const double f = (i == 2) ? csign : 1.0;
    put_sig(file, f * st.cell(i,0));
    put_sig(file, f * st.cell(i,1));
    put_sig(file, f * st.cell(i,2));
    file << std::endl;
  }


  // compute number of atoms for a given element
  const auto &elements = st.get_unique_elements();
  std::map<std::string, size_t> nelements;
  // Initialize the count for each element
  for (const auto& element : elements) {
    nelements[element.symbol] = 0;
  }
  // then count...
  for (const auto &atom : st) {
    nelements[atom.symbol]++; 
  }

  // write elements
  for (const auto& pair: nelements) {
    file << pair.first << " ";
  }
  file << std::endl;

  // write number of every elements
  for (const auto& pair: nelements) {
    file << pair.second << " ";
  }
  file << std::endl;

  file << "Cartesian" << std::endl;

  for (const auto& pair: nelements) {
    for (const auto &atom : st) {
      if (pair.first == atom.symbol) {
        put_sig(file, atom.position(0));
        put_sig(file, atom.position(1));
        put_sig(file, atom.position(2));
        file << std::endl;
      }
    }
  }

  file.close();
}
} // namespace mlip
} // namespace tadah
