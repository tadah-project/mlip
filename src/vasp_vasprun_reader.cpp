#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>

VaspVasprunReader::VaspVasprunReader(StructureDB& stdb)
: DatasetReader(stdb), stdb(stdb) {}

VaspVasprunReader::VaspVasprunReader(StructureDB& stdb, const std::string& filename)
: DatasetReader(stdb, filename), stdb(stdb) {
  read_data(filename);
}

VaspVasprunReader::~VaspVasprunReader() {
  delete xmlFile;
}

void VaspVasprunReader::read_data(const std::string& filename) {
  std::ifstream infile(filename);
  if (!infile.good()) {
    std::cerr << "Error: File " << filename << " cannot be opened or read." << std::endl;
    return;
  }

  try {
    xmlFile = new rx::file<>(filename.c_str());
    doc.parse<0>(xmlFile->data());
  } catch (std::exception &e) {
    std::cerr << "Error reading file: " << e.what() << std::endl;
  }

  _s.label = filename + " ";
}

void VaspVasprunReader::parse_data() {
  rx::xml_node<> *root_node = doc.first_node("modeling");
  if (!root_node) {
    std::cerr << "Root node not found." << std::endl;
    return;
  }

  extract_atom_types(root_node);
  extract_calculations(root_node);
}

int VaspVasprunReader::get_number_of_atoms() const {
  auto root_node = doc.first_node("modeling");
  if (!root_node) {
    std::cerr << "Root node not found." << std::endl;
    return 0;
  }

  auto atominfo_node = root_node->first_node("atominfo");
  if (!atominfo_node) {
    std::cerr << "atominfo node not found." << std::endl;
    return 0;
  }

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
  auto atominfo_node = root_node->first_node("atominfo");
  if (atominfo_node) {
    auto array_node = atominfo_node->first_node("array");
    if (array_node) {
      auto atoms_node = array_node->first_node("set");
      if (atoms_node) {
        for (auto rc_node = atoms_node->first_node("rc"); rc_node; rc_node = rc_node->next_sibling("rc")) {
          auto c_node = rc_node->first_node("c");
          if (c_node) {
            atom_types.push_back(c_node->value());
          }
        }
      }
    }
  }
}

void VaspVasprunReader::extract_calculations(rx::xml_node<> *root_node) {
  size_t counter=0;
  for (auto calculation_node = root_node->first_node("calculation");
  calculation_node; calculation_node = calculation_node->next_sibling("calculation")) {

    extract_total_energy(calculation_node);

    extract_stress_tensor(calculation_node);
    if (!stress_tensor_bool) {
      _s.stress.set_zero();
    }
    stress_tensor_bool = false;

    auto structure_node = calculation_node->first_node("structure");
    while (structure_node) {
      extract_basis_vectors_and_positions(structure_node);
      structure_node = structure_node->next_sibling("structure");
    }

    extract_forces(calculation_node);

    _s.label += "Structure " + std::to_string(++counter);
    stdb.add(_s);
    _s = Structure(); // reset
  }
}

void VaspVasprunReader::extract_total_energy(rx::xml_node<> *calculation_node) {
  auto energy_node = calculation_node->first_node("energy");
  if (energy_node) {
    for (auto energy_val_node = energy_node->first_node("i");
    energy_val_node; energy_val_node = energy_val_node->next_sibling("i")) {

      auto attribute = energy_val_node->first_attribute("name");
      if (attribute && std::string(attribute->value()) == "e_wo_entrp") {
        try {
          _s.energy = std::stod(energy_val_node->value());
        } catch (const std::invalid_argument&) {
          std::cerr << "Error converting energy value." << std::endl;
        }
      }
    }
  }
}

void VaspVasprunReader::extract_stress_tensor(rx::xml_node<> *calculation_node) {
  auto varray_node = calculation_node->first_node("varray");
  while (varray_node) {
    auto attribute = varray_node->first_attribute("name");
    if (attribute && std::string(attribute->value()) == "stress") {
      stress_tensor_bool = true;
      int r = 0;
      double x, y, z;
      for (auto v_node = varray_node->first_node("v");
      v_node; v_node = v_node->next_sibling("v")) {
        std::stringstream ss(v_node->value());
        if (ss >> x >> y >> z) {
          _s.stress(r, 0) = x;
          _s.stress(r, 1) = y;
          _s.stress(r, 2) = z;
        } else {
          std::cerr << "Error parsing stress tensor components." << std::endl;
        }
        r++;
      }
      _s.stress *= s_conv;
      break;
    }
    varray_node = varray_node->next_sibling("varray");
  }
}

void VaspVasprunReader::extract_basis_vectors_and_positions(rx::xml_node<> *structure_node) {
  auto crystal_node = structure_node->first_node("crystal");
  if (crystal_node) {
    auto basis_node = crystal_node->first_node("varray");
    while (basis_node) {
      auto attribute = basis_node->first_attribute("name");
      if (attribute && std::string(attribute->value()) == "basis") {
        size_t r = 0;
        double x, y, z;
        for (auto v_node = basis_node->first_node("v");
        v_node; v_node = v_node->next_sibling("v")) {
          std::stringstream ss(v_node->value());
          if (ss >> x >> y >> z) {
            _s.cell(r, 0) = x;
            _s.cell(r, 1) = y;
            _s.cell(r, 2) = z;
          } else {
            std::cerr << "Error parsing basis vector components." << std::endl;
          }
          r++;
        }
        break;
      }
      basis_node = basis_node->next_sibling("varray");
    }
  }

  auto positions_node = structure_node->first_node("varray");
  while (positions_node) {
    auto attribute = positions_node->first_attribute("name");
    if (attribute && std::string(attribute->value()) == "positions") {
      size_t index = 0;
      for (auto v_node = positions_node->first_node("v");
      v_node; v_node = v_node->next_sibling("v")) {
        double x, y, z;
        if (index < atom_types.size()) {
          std::stringstream ss(v_node->value());
          if (ss >> x >> y >> z) { // relative positions
            _s.add_atom(Atom(Element(atom_types[index]), x, y, z, 0, 0, 0));
            _s.atoms.back().position = _s.cell * _s.atoms.back().position;  // convert to abs
          } else {
            std::cerr << "Error parsing atom positions." << std::endl;
          }
        }
        index++;
      }
      break;
    }
    positions_node = positions_node->next_sibling("varray");
  }
}

void VaspVasprunReader::extract_forces(rx::xml_node<> *calculation_node) {
  auto forces_node = calculation_node->first_node("varray");
  while (forces_node) {
    auto attribute = forces_node->first_attribute("name");
    if (attribute && std::string(attribute->value()) == "forces") {
      size_t index = 0;
      double x, y, z;
      for (auto v_node = forces_node->first_node("v");
      v_node; v_node = v_node->next_sibling("v")) {
        std::stringstream ss(v_node->value());
        if (ss >> x >> y >> z) {
          _s.atoms[index].force[0] = x;
          _s.atoms[index].force[1] = y;
          _s.atoms[index].force[2] = z;
        } else {
          std::cerr << "Error parsing force components." << std::endl;
        }
        index++;
      }
      break;
    }
    forces_node = forces_node->next_sibling("varray");
  }
}

void VaspVasprunReader::print_summary() const {
  std::cout << get_summary();
}

std::string VaspVasprunReader::get_summary() const {
  return stdb.summary();
}
