#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure_readers/cif_reader.h>

#include <gemmi/cif.hpp>
#include <gemmi/smcif.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

void CifReader::read(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("CifReader::read: Cannot open " + path);
  }

  std::ostringstream buffer;
  buffer << ifs.rdbuf();
  parseCifContents(buffer.str());

  // Label the source
  m_structure.label = "CIF file: " + path;
}

Structure CifReader::getStructure() const { return m_structure; }

void CifReader::parseCifContents(const std::string &fileContents) {
  using namespace gemmi::cif;

  // Parses the raw CIF content into a gemmi Document
  Document doc = read_string(fileContents);
  if (doc.blocks.empty()) {
    throw std::runtime_error(
        "CifReader::parseCifContents: No data blocks found.");
  }

  // Assumes there is only one data block in the CIF
  Block &block = doc.sole_block();
  gemmi::SmallStructure struc = gemmi::make_small_structure_from_block(block);
  struc.check_spacegroup();

  auto sites = struc.get_all_unit_cell_sites();

  m_structure.atoms.reserve(sites.size());
  for (const auto &site : sites) {
    auto p = struc.cell.orthogonalize(site.fract);
    auto label = site.element.name();
    Element e = PeriodicTable().find_by_symbol(label);
    Atom a(e, p.x, p.y, p.z, 0, 0, 0);
    m_structure.atoms.push_back(a);
  }
  auto mat = struc.cell.orth.mat;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      m_structure.cell(i, j) = mat.a[i][j];
    }
  }
}
