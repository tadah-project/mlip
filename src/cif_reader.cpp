#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure_readers/cif_reader.h>

#include <gemmi/cif.hpp>
#include <gemmi/smcif.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
namespace tadah {
namespace mlip {

namespace {

// What gemmi expands the listed sites with: a tabulated space group, the
// operators of a symmetry-operator loop that is no tabulated setting, or
// nothing (P 1).
std::string symmetry_used(const gemmi::SmallStructure &struc) {
  if (struc.spacegroup) return "space group " + struc.spacegroup->xhm();
  if (!struc.cell.images.empty())
    return "the " + std::to_string(struc.cell.images.size() + 1) +
           " operators of the symmetry-operator loop (no tabulated space "
           "group)";
  return "no symmetry (P 1, the sites as listed)";
}

// gemmi's check_spacegroup() gives one finding per line; join them.
std::string one_line(const std::string &text) {
  std::istringstream in(text);
  std::string line, out;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (!out.empty()) out += "; ";
    out += line;
  }
  return out;
}

}  // namespace

void CifReader::read(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("CifReader::read: Cannot open " + path);
  }

  // Label the source
  m_structure.label = "CIF file: " + path;

  std::ostringstream buffer;
  buffer << ifs.rdbuf();
  parseCif(buffer.str(), path);

}

Structure CifReader::getStructure() const { return m_structure; }

void CifReader::parseCifContents(const std::string &fileContents) {
  parseCif(fileContents, "");
}

void CifReader::parseCif(const std::string &fileContents,
                         const std::string &path) {
  using namespace gemmi::cif;

  // Parses the raw CIF content into a gemmi Document
  Document doc = read_string(fileContents);
  if (doc.blocks.empty()) {
    throw std::runtime_error(
        "CifReader::parseCifContents: No data blocks found.");
  }

  // Assumes there is only one data block in the CIF
  Block &block = doc.sole_block();
  // Warnings and errors name the file, or the data block when there is none
  const std::string source =
      path.empty() ? "CIF text (data_" + block.name + ")" : "CIF file " + path;

  gemmi::SmallStructure struc = gemmi::make_small_structure_from_block(block);
  // gemmi looks the space group up in the order "S.H2": the operator loop,
  // then the Hall and H-M symbols, never the number. Without a loop it
  // reaches the symbols only because it swallows the std::length_error its
  // own "S." step throws on the empty operator list. Look them up here,
  // then the number (its standard setting); no record at all stays P 1.
  if (struc.spacegroup == nullptr && struc.symops.empty())
    struc.determine_and_set_spacegroup("H2N");

  // A partly occupied site (disorder or a vacancy) has no representation in
  // Tadah: two species sharing a site would become coincident atoms and a
  // vacancy a whole atom; an occupancy above 1 is not valid. The column is
  // read here, not from Site::occ, because gemmi takes a value that is not
  // a number as 1. An unknown occupancy ('?' or '.') is 1.
  enum { kLabel, kOcc };
  for (auto row : block.find("_atom_site_", {"label", "?occupancy"})) {
    if (!row.has2(kOcc)) continue;
    const std::string label = as_string(row[kLabel]);
    const std::string text = as_string(row[kOcc]);
    const double occ = as_number(text);
    if (std::isnan(occ)) {
      throw std::runtime_error(source + ": site " + label +
                               " has occupancy '" + text +
                               "', which is not a number.");
    }
    if (occ < 1.0 - 1e-6) {
      std::ostringstream msg;
      msg << source << ": site " << label << " has occupancy " << occ
          << "; Tadah has no representation for a partly occupied site "
             "(disorder or a vacancy): give an ordered structure, one atom "
             "per site.";
      throw std::runtime_error(msg.str());
    }
    if (occ > 1.0 + 1e-6) {
      std::ostringstream msg;
      msg << source << ": site " << label << " has occupancy " << occ
          << ", above 1, which is not a valid occupancy.";
      throw std::runtime_error(msg.str());
    }
  }

  // A record that contradicts itself (or names nothing gemmi knows) is
  // reported; the structure is built with what gemmi found first.
  const std::string diagnostic = struc.check_spacegroup();
  if (!diagnostic.empty()) {
    std::cerr << "WARNING: " << source
              << ": the symmetry record is inconsistent (gemmi: \""
              << one_line(diagnostic) << "\"); the unit cell is built with "
              << symmetry_used(struc) << "." << std::endl;
  }

  auto sites = struc.get_all_unit_cell_sites();

  auto volume = struc.cell.volume;
  auto symmetry = struc.spacegroup_hm;
  auto elements = struc.atom_types;
  m_structure.label += " | Symmetry: " + symmetry;
  m_structure.label += " | Volume: " + std::to_string(volume);

  m_structure.atoms.reserve(sites.size());
  for (const auto &site : sites) {
    auto p = struc.cell.orthogonalize(site.fract);
    auto label = site.element.name();
    tadah::core::Element e = tadah::core::PeriodicTable().find_by_symbol(label);
    Atom a(e, p.x, p.y, p.z, 0, 0, 0);
    m_structure.atoms.push_back(a);
  }
  // gemmi's orthogonalisation matrix maps fractional to Cartesian
  // coordinates (r = orth * f), so its COLUMNS are the lattice vectors.
  // Structure::cell stores lattice vector a_i as row i: transpose.
  auto mat = struc.cell.orth.mat;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      m_structure.cell(i, j) = mat.a[j][i];
    }
  }
}
} // namespace mlip
} // namespace tadah
