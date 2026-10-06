#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/castep_cell_writer.h>
namespace tadah {
namespace mlip {

CastepCellWriter::CastepCellWriter(StructureDB& db) : StructureWriter(db) {}

void CastepCellWriter::write_data(const std::string& outFile, const size_t i) {

  check(i);

  std::ofstream file(outFile);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + outFile);
  }

  const Structure &st = stdb(i);

  // write label
  file << "# " << st.label << std::endl;

  // write cell
  file << "%BLOCK LATTICE_CART" << std::endl;
  file << "ANG" << std::endl;
  for (int i=0; i<3; ++i) {
    file << ' ' << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,0);
    file << ' ' << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,1);
    file << ' ' << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,2);
    file << std::endl;
  }
  file << "%ENDBLOCK LATTICE_CART" << std::endl;

  file << std::endl;

  file << "%BLOCK POSITIONS_ABS" << std::endl;
  for (const auto &atom : st) {
    file << ' ' << std::right << std::fixed << std::setw(w)
      << atom.symbol;
    file << ' ' << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(0);
    file << ' ' << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(1);
    file << ' ' << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(2);
    file << std::endl;
  }
  file << "%ENDBLOCK POSITIONS_ABS" << std::endl;
  file.close();
  // A write can fail after the file opened (a full disk).
  if (file.fail()) throw std::runtime_error("Could not write the file: " + outFile);
}
} // namespace mlip
} // namespace tadah
