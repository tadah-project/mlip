#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>
#include <tadah/mlip/dataset_readers/missing_stress.h>

#include <tadah/core/periodic_table.h>
#include <tadah/core/utils.h>

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace tadah {
namespace mlip {

namespace {

/* The <v> rows of a varray node. */
std::vector<rx::xml_node<> *> rows_of(rx::xml_node<> *varray) {
  std::vector<rx::xml_node<> *> rows;
  for (auto v = varray->first_node("v"); v; v = v->next_sibling("v"))
    rows.push_back(v);
  return rows;
}

/* The three numbers of a <v> row (as the stream reads VASP's fixed-width
 * columns, which may touch: '1.0-2.0'); false when the row holds anything
 * else. */
bool three_numbers(const rx::xml_node<> *v, double &x, double &y, double &z) {
  std::istringstream ss(v->value());
  return static_cast<bool>(ss >> x >> y >> z) && (ss >> std::ws).eof();
}

/* The first child varray of `parent` named `name`, or null. */
rx::xml_node<> *varray_named(rx::xml_node<> *parent, const std::string &name) {
  for (auto n = parent->first_node("varray"); n; n = n->next_sibling("varray")) {
    auto a = n->first_attribute("name");
    if (a && std::string(a->value()) == name) return n;
  }
  return nullptr;
}

std::string trimmed(const rx::xml_node<> *v) {
  return tadah::core::trim(std::string(v->value()));
}

}  // namespace

VaspVasprunReader::VaspVasprunReader(StructureDB& stdb)
: DatasetReader(stdb), stdb(stdb) {}

VaspVasprunReader::VaspVasprunReader(StructureDB& stdb, const std::string& filename)
: DatasetReader(stdb, filename), stdb(stdb) {
  read_data(filename);
}

VaspVasprunReader::~VaspVasprunReader() = default;

std::string VaspVasprunReader::where() const {
  return "VASP vasprun.xml " + filename_ + ": ";
}

std::string VaspVasprunReader::where_calculation() const {
  return where() + "calculation " + std::to_string(calculation_) + ": ";
}

namespace {

constexpr int kParseFlags = rx::parse_validate_closing_tags;

/* True when a <calculation> element opens in `text` at or after `from`:
 * '<calculation' followed by '>', a blank or '/'. */
bool calculation_opens(const std::string &text, std::size_t from) {
  const std::string open = "<calculation";
  for (std::size_t p = text.find(open, from); p != std::string::npos;
       p = text.find(open, p + open.size())) {
    const std::size_t q = p + open.size();
    if (q == text.size() || text[q] == '>' || text[q] == '/' ||
        std::isspace(static_cast<unsigned char>(text[q])))
      return true;
  }
  return false;
}

/* `text` as the null-terminated buffer rapidxml parses in place. */
std::vector<char> xml_buffer(const std::string &text) {
  std::vector<char> b(text.begin(), text.end());
  b.push_back('\0');
  return b;
}

}  // namespace

void VaspVasprunReader::read_data(const std::string& filename) {
  filename_ = filename;
  std::ifstream infile(filename, std::ios::binary);
  if (!infile.good())
    throw std::runtime_error(where() + "cannot open the file");
  std::ostringstream bytes;
  bytes << infile.rdbuf();
  if (infile.bad())
    throw std::runtime_error(where() + "cannot read the file");
  const std::string text = bytes.str();

  doc.clear();
  ends_inside_ = 0;
  ends_after_ = 0;
  ends_before_first_ = false;
  buffer_ = xml_buffer(text);
  try {
    doc.parse<kParseFlags>(buffer_.data());
  } catch (const rx::parse_error &e) {
    // VASP writes the file as it runs: a run stopped, or still running,
    // leaves it without its end, and the parser stops at the end of the
    // text. Its complete calculations are read; the one it ends inside, if a
    // calculation opens after the last complete one, is skipped with a
    // warning (parse_data). An error before the end is a file that is not
    // well-formed.
    const char *at = e.where<char>();
    const bool at_end =
        at != nullptr && at >= buffer_.data() &&
        static_cast<std::size_t>(at - buffer_.data()) >= text.size();
    if (!at_end)
      throw std::runtime_error(where() + "not well-formed XML (" + e.what() + ")");
    const std::string close = "</calculation>";
    const std::size_t last = text.rfind(close);
    std::size_t complete = 0;
    for (std::size_t p = text.find(close); p != std::string::npos;
         p = text.find(close, p + close.size()))
      ++complete;
    doc.clear();
    if (last == std::string::npos) {
      // No calculation is complete: nothing to read.
      buffer_.clear();
      if (!calculation_opens(text, 0))
        ends_before_first_ = true;
      else
        ends_inside_ = 1;
    } else {
      buffer_ = xml_buffer(text.substr(0, last + close.size()) +
                           "\n</modeling>\n");
      // The text after the last complete calculation: the start of the
      // next one, or what VASP writes after the last (the final structure).
      if (calculation_opens(text, last + close.size()))
        ends_inside_ = complete + 1;
      else
        ends_after_ = complete;
      try {
        doc.parse<kParseFlags>(buffer_.data());
      } catch (const rx::parse_error &e2) {
        throw std::runtime_error(where() + "not well-formed XML (" +
                                 e2.what() + ")");
      }
    }
  }

  _s.label = filename + " ";
}

void VaspVasprunReader::parse_data() {
  if (buffer_.empty()) {
    // The file ends before any calculation is complete.
    std::cerr << "WARNING: " << where() << "the file ends "
              << (ends_before_first_ ? "before its first calculation"
                                     : "inside calculation 1")
              << " (a run stopped while writing it)"
              << (ends_before_first_ ? "; " : ", which was skipped; ")
              << "0 structure(s) read." << std::endl;
    return;
  }
  rx::xml_node<> *root_node = doc.first_node("modeling");
  if (!root_node)
    throw std::runtime_error(where() + "no <modeling> root element");

  extract_atom_types(root_node);
  const std::size_t before = stdb.size();
  extract_calculations(root_node);
  if (ends_inside_ > 0)
    std::cerr << "WARNING: " << where() << "the file ends inside calculation "
              << ends_inside_ << " (a run stopped while writing it), which was "
              << "skipped; " << stdb.size() - before << " structure(s) read."
              << std::endl;
  else if (ends_after_ > 0)
    std::cerr << "WARNING: " << where() << "the file ends after calculation "
              << ends_after_ << ", without its closing </modeling> tag (a run "
              << "stopped, or still running); no calculation was skipped: "
              << stdb.size() - before << " structure(s) read." << std::endl;
}

int VaspVasprunReader::get_number_of_atoms() const {
  auto root_node = doc.first_node("modeling");
  if (!root_node)
    throw std::runtime_error(where() + "no <modeling> root element");

  auto atominfo_node = root_node->first_node("atominfo");
  if (!atominfo_node)
    throw std::runtime_error(where() + "no <atominfo> element");

  int count = 0;
  auto array_node = atominfo_node->first_node("array");
  if (array_node) {
    auto atoms_node = array_node->first_node("set");
    if (atoms_node) {
      for (auto rc_node = atoms_node->first_node("rc"); rc_node; rc_node = rc_node->next_sibling("rc")) {
        count++;
      }
    }
  }
  return count;
}

void VaspVasprunReader::extract_atom_types(rx::xml_node<> *root_node) {
  atom_types.clear();
  elements_.clear();
  auto atominfo_node = root_node->first_node("atominfo");
  if (atominfo_node) {
    auto array_node = atominfo_node->first_node("array");
    if (array_node) {
      auto atoms_node = array_node->first_node("set");
      if (atoms_node) {
        for (auto rc_node = atoms_node->first_node("rc"); rc_node; rc_node = rc_node->next_sibling("rc")) {
          auto c_node = rc_node->first_node("c");
          if (!c_node)
            throw std::runtime_error(where() + "atom " +
                                     std::to_string(atom_types.size() + 1) +
                                     " of <atominfo> has no element name");
          // VASP pads the name to two characters ('W ', 'O ').
          const std::string name = trimmed(c_node);
          try {
            elements_.push_back(tadah::core::PeriodicTable::find_by_symbol(name));
          } catch (const std::exception &) {
            throw std::runtime_error(where() + "atom " +
                                     std::to_string(atom_types.size() + 1) +
                                     " of <atominfo>, '" + name +
                                     "', is not an element symbol");
          }
          atom_types.push_back(name);
        }
      }
    }
  }
  if (atom_types.empty())
    throw std::runtime_error(where() + "no atoms in <atominfo>");
}

void VaspVasprunReader::extract_calculations(rx::xml_node<> *root_node) {
  size_t counter=0, without_stress=0, without_forces=0;
  for (auto calculation_node = root_node->first_node("calculation");
  calculation_node; calculation_node = calculation_node->next_sibling("calculation")) {

    calculation_ = counter + 1;
    extract_total_energy(calculation_node);

    extract_stress_tensor(calculation_node);
    if (!stress_tensor_bool) {
      // No stress tensor: zero stress and stress priority 0, so that a
      // STRESS fit ignores it.
      _s.stress.set_zero();
      _s.sweight = 0.0;
      ++without_stress;
    }
    stress_tensor_bool = false;

    // One geometry per calculation.
    size_t structures = 0;
    for (auto n = calculation_node->first_node("structure"); n;
         n = n->next_sibling("structure"))
      ++structures;
    if (structures != 1)
      throw std::runtime_error(where_calculation() + "has " +
                               std::to_string(structures) +
                               " <structure> elements, not 1");
    extract_basis_vectors_and_positions(calculation_node->first_node("structure"));

    extract_forces(calculation_node);
    if (!forces_found_) {
      // No forces: zero forces and force priority 0, so that a FORCE fit
      // ignores them.
      for (auto &a : _s.atoms) a.force = tadah::core::Vec3d(0.0, 0.0, 0.0);
      _s.fweight = 0.0;
      ++without_forces;
    }
    forces_found_ = false;

    _s.label += "Structure " + std::to_string(++counter);
    stdb.add(_s);
    _s = Structure(); // reset
  }
  log_missing_stress("VASP vasprun.xml " + filename_, without_stress, counter);
  log_missing_forces("VASP vasprun.xml " + filename_, without_forces, counter);
}

void VaspVasprunReader::extract_total_energy(rx::xml_node<> *calculation_node) {
  bool found = false;
  auto energy_node = calculation_node->first_node("energy");
  if (energy_node) {
    for (auto energy_val_node = energy_node->first_node("i");
    energy_val_node; energy_val_node = energy_val_node->next_sibling("i")) {

      // The free energy F, whose gradient the forces and the stress are.
      // (The calculation-level 'e_wo_entrp' holds energy(sigma->0) in VASP
      // 5.4.4, a labelling bug; 'e_fr_energy' is right in every version.)
      auto attribute = energy_val_node->first_attribute("name");
      if (attribute && std::string(attribute->value()) == "e_fr_energy") {
        const std::string value = energy_val_node->value();
        std::istringstream iss(value);
        double e;
        std::string rest;
        if (!(iss >> e) || (iss >> rest))
          throw std::runtime_error(
            "VASP vasprun.xml " + filename_ + ": calculation " +
            std::to_string(calculation_) +
            ": the calculation-level <i name=\"e_fr_energy\"> value '" +
            value + "' is not a number");
        _s.energy = e;
        found = true;
      }
    }
  }
  if (!found)
    throw std::runtime_error(
      "VASP vasprun.xml " + filename_ + ": calculation " +
      std::to_string(calculation_) +
      " has no calculation-level <i name=\"e_fr_energy\"> (the free energy)");
}

void VaspVasprunReader::extract_stress_tensor(rx::xml_node<> *calculation_node) {
  rx::xml_node<> *varray = varray_named(calculation_node, "stress");
  if (!varray) return;
  const std::vector<rx::xml_node<> *> rows = rows_of(varray);
  if (rows.size() != 3)
    throw std::runtime_error(where_calculation() + "the stress varray has " +
                             std::to_string(rows.size()) + " rows, not 3");
  for (size_t r = 0; r < 3; ++r) {
    double x, y, z;
    if (!three_numbers(rows[r], x, y, z)) {
      // A value VASP could not print ('*****') is no stress tensor.
      std::cerr << "WARNING: " << where_calculation() << "the stress row "
                << r + 1 << ", '" << trimmed(rows[r])
                << "', cannot be read; the calculation is stored without "
                   "stress" << std::endl;
      _s.stress.set_zero();
      return;
    }
    _s.stress(r, 0) = x;
    _s.stress(r, 1) = y;
    _s.stress(r, 2) = z;
  }
  // VASP's tensor is pressure-signed (positive under compression);
  // Tadah stores the Cauchy stress, positive under tension.
  _s.stress *= -s_conv;
  stress_tensor_bool = true;
}

void VaspVasprunReader::extract_basis_vectors_and_positions(rx::xml_node<> *structure_node) {
  auto crystal_node = structure_node->first_node("crystal");
  rx::xml_node<> *basis = crystal_node ? varray_named(crystal_node, "basis") : nullptr;
  if (!basis)
    throw std::runtime_error(where_calculation() +
                             "the <structure> has no <crystal> basis varray");
  const std::vector<rx::xml_node<> *> vectors = rows_of(basis);
  if (vectors.size() != 3)
    throw std::runtime_error(where_calculation() + "the basis varray has " +
                             std::to_string(vectors.size()) + " rows, not 3");
  for (size_t r = 0; r < 3; ++r) {
    double x, y, z;
    if (!three_numbers(vectors[r], x, y, z))
      throw std::runtime_error(where_calculation() + "basis vector " +
                               std::to_string(r + 1) + ", '" +
                               trimmed(vectors[r]) +
                               "', is not three numbers");
    _s.cell(r, 0) = x;
    _s.cell(r, 1) = y;
    _s.cell(r, 2) = z;
  }

  rx::xml_node<> *positions = varray_named(structure_node, "positions");
  if (!positions)
    throw std::runtime_error(where_calculation() +
                             "the <structure> has no positions varray");
  const std::vector<rx::xml_node<> *> rows = rows_of(positions);
  if (rows.size() != atom_types.size())
    throw std::runtime_error(where_calculation() + "the positions varray has " +
                             std::to_string(rows.size()) + " rows for the " +
                             std::to_string(atom_types.size()) +
                             " atoms of <atominfo>");
  for (size_t index = 0; index < rows.size(); ++index) {
    double x, y, z;
    if (!three_numbers(rows[index], x, y, z))  // relative positions
      throw std::runtime_error(where_calculation() + "the position of atom " +
                               std::to_string(index + 1) + ", '" +
                               trimmed(rows[index]) +
                               "', is not three numbers");
    _s.add_atom(Atom(elements_[index], x, y, z, 0, 0, 0));
    // Rows-as-lattice-vectors: r = cell^T * frac.
    _s.atoms.back().position = _s.cell.transpose() * _s.atoms.back().position;
  }
}

void VaspVasprunReader::extract_forces(rx::xml_node<> *calculation_node) {
  forces_found_ = false;
  rx::xml_node<> *varray = varray_named(calculation_node, "forces");
  if (!varray) return;
  const std::vector<rx::xml_node<> *> rows = rows_of(varray);
  if (rows.size() != _s.atoms.size())
    throw std::runtime_error(where_calculation() + "the forces varray has " +
                             std::to_string(rows.size()) + " rows for the " +
                             std::to_string(_s.atoms.size()) +
                             " atoms of <atominfo>");
  for (size_t index = 0; index < rows.size(); ++index) {
    double x, y, z;
    if (!three_numbers(rows[index], x, y, z)) {
      // A value VASP could not print ('*****') is no forces.
      std::cerr << "WARNING: " << where_calculation() << "the force of atom "
                << index + 1 << ", '" << trimmed(rows[index])
                << "', cannot be read; the calculation is stored without "
                   "forces" << std::endl;
      return;
    }
    _s.atoms[index].force[0] = x;
    _s.atoms[index].force[1] = y;
    _s.atoms[index].force[2] = z;
  }
  forces_found_ = true;
}

void VaspVasprunReader::print_summary() const {
  std::cout << get_summary();
}

std::string VaspVasprunReader::get_summary() const {
  return stdb.summary();
}
} // namespace mlip
} // namespace tadah
