#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_writers/castep_cell_writer.h>

CastepCellWriter::CastepCellWriter(StructureDB& db) : DatasetWriter(db) {}

void CastepCellWriter::write_data(const std::string& filename, const size_t i) {

  if (i >= stdb.size()) {
    throw std::out_of_range("Index i is out of range.");
  }
    
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + filename);
  }

  const Structure &st = stdb(i);

  // write label
  file << "# " << st.label << std::endl;

  // write cell
  file << "%BLOCK LATTICE_CART" << std::endl;
  file << "ANG" << std::endl;
  for (int i=0; i<3; ++i) {
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,0);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,1);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << st.cell(i,2);
    file << std::endl;
  }
  file << "%ENDBLOCK LATTICE_CART" << std::endl;

  file << std::endl;

  file << "%BLOCK POSITIONS_ABS" << std::endl;
  for (const auto &atom : st) {
    file << std::right << std::fixed << std::setw(w)
      << atom.symbol;
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(0);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(1);
    file << std::right << std::fixed << std::setw(w)
      << std::setprecision(p) << atom.position(2);
    file << std::endl;
  }
  file << "%ENDBLOCK POSITIONS_ABS	" << std::endl;
  file.close();
}
