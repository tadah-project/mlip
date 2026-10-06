#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure_readers/cif_reader.h>

#include <gemmi/cif.hpp>
#include <gemmi/smcif.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
namespace tadah {
namespace mlip {

namespace {

// What gemmi expands the listed sites with: a tabulated space group, the
// operators of a symmetry-operator loop or a Hall symbol that is no
// tabulated setting, or nothing (P 1).
std::string symmetry_used(const gemmi::SmallStructure &struc) {
  if (struc.spacegroup) return "space group " + struc.spacegroup->xhm();
  if (!struc.cell.images.empty())
    return "the " + std::to_string(struc.cell.images.size() + 1) +
           " operators of " +
           (struc.symops.empty()
                ? "the Hall symbol '" + struc.spacegroup_hall + "'"
                : std::string("the symmetry-operator loop")) +
           " (no tabulated space group)";
  return "no symmetry (P 1, the sites as listed)";
}

void trim(std::string &s) {
  const std::size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) {
    s.clear();
    return;
  }
  s = s.substr(b, s.find_last_not_of(" \t") - b + 1);
}

// The H-M symbol in the spelling gemmi's table lookup reads. CIF writers
// spell one symbol in several ways: subscripts written with '_' ('P 2_1/c',
// spglib) or in parentheses ('P 4(2)/n m c'), a trailing ' (No. 14)', and
// ICSD's origin-choice suffix ' S' or ' Z' (origin choice 1 or 2), which
// gemmi reads as ':1' or ':2'.
std::string normalised_hm(std::string hm) {
  static const std::regex number_note(R"(\s*\(\s*[Nn][Oo]\.?\s*\d+\s*\)$)");
  static const std::regex subscript(R"((\d)\((\d)\))");
  trim(hm);
  hm = std::regex_replace(hm, number_note, "");
  hm = std::regex_replace(hm, subscript, "$1$2");
  hm.erase(std::remove(hm.begin(), hm.end(), '_'), hm.end());
  trim(hm);
  if (hm.size() > 2 && hm[hm.size() - 2] == ' ') {
    const char c = hm.back();
    if (c == 'S' || c == 's')
      hm = hm.substr(0, hm.size() - 2) + ":1";
    else if (c == 'Z' || c == 'z')
      hm = hm.substr(0, hm.size() - 2) + ":2";
  }
  return hm;
}

// The tabulated settings of the group of @p sg with its centring letter:
// those a number alone leaves open once the cell is known (hexagonal or
// rhombohedral axes, the origin choice, the cell choice of a monoclinic
// group, the axes of an orthorhombic one). gemmi also lists P 1, P -1 and
// a few others in centred cells (A 1, ..., F 4/m m m), which describe
// another cell, not the given one.
int settings_like(const gemmi::SpaceGroup &sg) {
  int n = 0;
  for (const gemmi::SpaceGroup &t : gemmi::spacegroup_tables::main)
    n += t.number == sg.number && t.hm[0] == sg.hm[0];
  return n;
}

// A monoclinic group (IT 3 to 15) is described in the setting its unique
// axis names, so the two cell angles that do not involve that axis are 90
// degrees (alpha is the angle between b and c, beta between a and c, gamma
// between a and b). In a cell whose angles say otherwise, the operators
// place the images of a site where that crystal has none.
void require_monoclinic_cell(const gemmi::SmallStructure &struc,
                             const std::string &source,
                             const std::string &given_by) {
  const gemmi::SpaceGroup *sg = struc.spacegroup;
  if (!sg || sg->number < 3 || sg->number > 15) return;
  const char *q = sg->qualifier;
  if (*q == '-') ++q;
  const gemmi::UnitCell &c = struc.cell;
  double first = 0.0, second = 0.0;
  const char *needs = nullptr;
  switch (*q) {
    case 'a': first = c.beta; second = c.gamma; needs = "beta = gamma"; break;
    case 'b': first = c.alpha; second = c.gamma; needs = "alpha = gamma"; break;
    case 'c': first = c.alpha; second = c.beta; needs = "alpha = beta"; break;
    default:
      throw std::logic_error("CifReader: monoclinic space group " + sg->xhm() +
                             " has no unique axis in gemmi's table");
  }
  constexpr double tol = 1e-3;  // degrees
  if (std::fabs(first - 90.0) <= tol && std::fabs(second - 90.0) <= tol)
    return;
  using tadah::core::format_roundtrip;
  throw std::runtime_error(
      source + ": the space group " + sg->xhm() + " (from " + given_by +
      ") is monoclinic with unique axis " + *q + ", which needs " + needs +
      " = 90 degrees (within 1e-3), but the cell has alpha " +
      format_roundtrip(c.alpha) + ", beta " + format_roundtrip(c.beta) +
      ", gamma " + format_roundtrip(c.gamma) +
      "; give the full H-M symbol of the setting the cell is in (e.g. "
      "'P 1 1 21/m' for unique axis c), or its symmetry-operator loop.");
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
  // The H-M symbol as gemmi's table lookup reads it, for every use below:
  // the fallback from an operator loop that forms no group, the lookup
  // without a loop and gemmi's consistency check.
  const std::string hm_as_written = struc.spacegroup_hm;
  struc.spacegroup_hm = normalised_hm(struc.spacegroup_hm);
  std::string given_by;  // for the monoclinic check's message
  bool number_has_choice = false;
  if (!struc.symops.empty()) {
    // gemmi looked the space group up in the order "S.H2": the operator
    // loop, then (when the loop forms no group) the Hall and H-M symbols.
    // Repeat it with the normalised H-M symbol.
    if (struc.spacegroup_hm != hm_as_written)
      struc.determine_and_set_spacegroup("S.H2");
    gemmi::GroupOps unused;
    const gemmi::SpaceGroup *from_loop = nullptr;
    try {
      from_loop = struc.determine_spacegroup_from('S', unused);
    } catch (const std::exception &) {
      // an operator gemmi cannot parse: the group came from a symbol
    }
    if (struc.spacegroup == from_loop)
      given_by = "the symmetry-operator loop";
    else
      given_by = "the " +
                 (struc.spacegroup_hall.empty()
                      ? "H-M symbol '" + hm_as_written + "'"
                      : "Hall or H-M symbol") +
                 ", as the operator loop forms no group";
  } else {
    // Without a loop gemmi reaches the symbols only because it swallows the
    // std::length_error its own "S." step throws on the empty operator
    // list, and it never uses the number. Look the space group up here:
    // the Hall symbol, else the H-M symbol, else the number (each in the
    // setting it names, or else origin choice 2 and hexagonal or
    // rhombohedral axes from the cell angles); no record at all is P 1.
    // A symbol that cannot be read is refused: the setting it means cannot
    // be told, and the number would be a guess.
    struc.spacegroup = nullptr;
    const auto cannot_read = [&](const std::string &kind,
                                 const std::string &symbol,
                                 const std::string &why) {
      return std::runtime_error(
          source + ": cannot read the " + kind + " symbol '" + symbol +
          "': " + why + "; give " +
          (kind == "Hall" ? "a Hall symbol in standard form (e.g. '-F 4vw "
                            "2vw 3'), the H-M symbol"
                          : "it in a standard form (e.g. 'F d -3 m:2', or "
                            "'F d -3 m Z' for origin choice 2)") +
          ", or the symmetry-operator loop.");
    };
    if (!struc.spacegroup_hall.empty()) {
      given_by = "the Hall symbol '" + struc.spacegroup_hall + "'";
      gemmi::GroupOps ops;
      try {
        ops = gemmi::symops_from_hall(struc.spacegroup_hall.c_str());
      } catch (const std::exception &e) {
        throw cannot_read("Hall", struc.spacegroup_hall,
                          std::string("gemmi: ") + e.what());
      }
      struc.spacegroup = gemmi::find_spacegroup_by_ops(ops);
      if (struc.spacegroup)
        struc.setup_cell_images();
      else  // a group in no tabulated setting: its own operators
        struc.cell.set_cell_images_from_groupops(ops);
    } else if (!struc.spacegroup_hm.empty()) {
      given_by = "the H-M symbol '" + hm_as_written + "'";
      // gemmi's name lookup takes a number as a CCP4 code, which is the IT
      // number only up to 230 (1003 is P 1 1 2): a number is no H-M symbol.
      if (std::isdigit(static_cast<unsigned char>(struc.spacegroup_hm[0])))
        throw cannot_read("space-group", hm_as_written,
                          "it is a number, not an H-M symbol (the number "
                          "goes in _space_group_IT_number)");
      struc.spacegroup = gemmi::find_spacegroup_by_name(
          struc.spacegroup_hm, struc.cell.alpha, struc.cell.gamma, "2");
      if (!struc.spacegroup)
        throw cannot_read("space-group", hm_as_written,
                          "it names no space group gemmi knows");
      struc.setup_cell_images();
    } else if (struc.spacegroup_number != 0) {
      const int n = struc.spacegroup_number;
      given_by = "the number " + std::to_string(n);
      if (n < 1 || n > 230)
        throw std::runtime_error(
            source + ": the space-group number " + std::to_string(n) +
            " is not one of the 230 space groups (1 to 230); give the H-M "
            "symbol or the symmetry-operator loop.");
      // gemmi's own number lookup takes the first tabulated setting,
      // whatever the cell (R -3 m:H for a rhombohedral cell): resolve the
      // number through the H-M symbol of its reference setting, as the
      // symbol would be.
      const gemmi::SpaceGroup &ref = gemmi::get_spacegroup_reference_setting(n);
      const gemmi::SpaceGroup *sg = gemmi::find_spacegroup_by_name(
          ref.hm, struc.cell.alpha, struc.cell.gamma, "2");
      struc.spacegroup = sg ? sg : &ref;
      struc.setup_cell_images();
      number_has_choice = settings_like(*struc.spacegroup) > 1;
    } else {
      struc.setup_cell_images();
    }
  }
  require_monoclinic_cell(struc, source, given_by);
  if (number_has_choice)
    std::cerr << "WARNING: " << source
              << ": the space group is given only by its number ("
              << struc.spacegroup_number
              << "), which has more than one setting; the unit cell is "
                 "built with space group "
              << struc.spacegroup->xhm()
              << ", its standard setting (hexagonal or rhombohedral axes "
                 "from the cell angles, origin choice 2 where the group has "
                 "two)."
              << std::endl;

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
      msg << source << ": site " << label << " has occupancy " << text
          << "; Tadah has no representation for a partly occupied site "
             "(disorder or a vacancy): give an ordered structure, one atom "
             "per site.";
      throw std::runtime_error(msg.str());
    }
    if (occ > 1.0 + 1e-6) {
      std::ostringstream msg;
      msg << source << ": site " << label << " has occupancy " << text
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
