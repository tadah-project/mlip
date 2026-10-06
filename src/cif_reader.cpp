#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure_readers/cif_reader.h>

#include <gemmi/cif.hpp>
#include <gemmi/smcif.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <utility>
#include <regex>
#include <sstream>
#include <stdexcept>
namespace tadah {
namespace mlip {

namespace {

// Where the symmetry the cell is built with comes from.
struct SymmetrySource {
  enum Kind { None, Loop, Hall, HM, Number } kind = None;
  std::string record;  // "the Hall symbol '-P 2yb'", ...
  std::string note;    // " (the operator loop forms no group)" or ""
  // The operators, when they are in no tabulated setting
  std::optional<gemmi::GroupOps> own_ops;
};

// What gemmi expands the listed sites with: a tabulated space group, the
// operators of a loop or a Hall symbol that is in no tabulated setting, or
// nothing (P 1).
std::string symmetry_used(const gemmi::SmallStructure &struc,
                          const SymmetrySource &src) {
  if (struc.spacegroup) return "space group " + struc.spacegroup->xhm();
  if (src.own_ops)
    return "the " + std::to_string(src.own_ops->order()) + " operators of " +
           src.record + " (no tabulated space group)";
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
  // The axes of an R group written as a suffix ('R -3 m H', ICSD; 'R3R'),
  // which gemmi reads as ':H' / ':R': spelled so, to be seen as named.
  if (hm.size() > 1 && (hm[0] == 'R' || hm[0] == 'r') &&
      hm.find(':') == std::string::npos) {
    const char c = static_cast<char>(
        std::toupper(static_cast<unsigned char>(hm.back())));
    if (c == 'H' || c == 'R') {
      hm.pop_back();
      trim(hm);
      hm += std::string(":") + c;
    }
  }
  return hm;
}

// The unique axis ('a', 'b' or 'c') of a monoclinic group (IT 3 to 15) of
// gemmi's table, or 0 for any other group.
char monoclinic_axis(const gemmi::SpaceGroup &sg) {
  if (sg.number < 3 || sg.number > 15) return 0;
  const char *q = sg.qualifier;
  if (*q == '-') ++q;
  if (*q != 'a' && *q != 'b' && *q != 'c')
    throw std::logic_error("CifReader: monoclinic space group " + sg.xhm() +
                           " has no unique axis in gemmi's table");
  return *q;
}

// The two cell angles between unique axis @p axis and the other axes (for
// b, alpha, between b and c, and gamma, between a and b), which a
// monoclinic group needs to be 90 degrees; the third angle is free.
std::pair<double, double> angles_at(const gemmi::UnitCell &c, char axis) {
  switch (axis) {
    case 'a': return {c.beta, c.gamma};
    case 'b': return {c.alpha, c.gamma};
    default: return {c.alpha, c.beta};
  }
}

constexpr double kAngleTol = 1e-3;  // degrees

// The axes a group's operators fix, which the cell angles must fit: a
// monoclinic group's unique axis (the angles between it and the other two
// axes are 90 degrees; the third is free), an R group's hexagonal axes
// (alpha = beta = 90, gamma = 120) or rhombohedral axes (alpha = beta =
// gamma). Other groups are not checked (Free).
struct Axes {
  enum Kind { Free, Monoclinic, Hexagonal, Rhombohedral } kind = Free;
  char unique = 0;  // Monoclinic: 'a', 'b' or 'c'
};

Axes axes_of(const gemmi::SpaceGroup &sg) {
  if (const char axis = monoclinic_axis(sg)) return {Axes::Monoclinic, axis};
  if (sg.hm[0] == 'R' && sg.ext == 'H') return {Axes::Hexagonal, 0};
  if (sg.hm[0] == 'R' && sg.ext == 'R') return {Axes::Rhombohedral, 0};
  return {};
}

bool fits_cell(const gemmi::UnitCell &c, const Axes &a) {
  const auto near = [](double x, double y) {
    return std::fabs(x - y) <= kAngleTol;
  };
  switch (a.kind) {
    case Axes::Monoclinic: {
      const auto [first, second] = angles_at(c, a.unique);
      return near(first, 90.0) && near(second, 90.0);
    }
    case Axes::Hexagonal:
      return near(c.alpha, 90.0) && near(c.beta, 90.0) &&
             near(c.gamma, 120.0);
    case Axes::Rhombohedral:
      return near(c.alpha, c.beta) && near(c.beta, c.gamma) &&
             near(c.alpha, c.gamma);
    case Axes::Free:
      break;
  }
  return true;
}

// The ITA settings of the group of @p sg that cell @p c could be in, @p sg
// included: another origin choice (the groups with two) or another choice
// of axes (a permutation, a monoclinic unique axis or cell choice) that
// fits the cell, each a change of basis with determinant +-1 and a setting
// code (gemmi's qualifier and ext) of its own. gemmi's table also lists
// groups in larger centred cells (A 1, ..., F 4/m m m), R groups on both
// hexagonal and rhombohedral axes (other cells, which the cell itself
// tells apart), and CCP4's extra origins of 18, 20, 21, 22, 23, 94 and 197
// under the setting code of the standard setting: none of them counts.
int settings_of_the_cell(const gemmi::SpaceGroup &sg,
                         const gemmi::UnitCell &c) {
  const int det = std::abs(sg.basisop().det_rot());
  int n = 1;
  for (const gemmi::SpaceGroup &t : gemmi::spacegroup_tables::main)
    if (t.number == sg.number && &t != &sg &&
        (t.ext != sg.ext || std::strcmp(t.qualifier, sg.qualifier) != 0) &&
        std::abs(t.basisop().det_rot()) == det && fits_cell(c, axes_of(t)))
      ++n;
  return n;
}

// The axes of operators in no tabulated setting (an origin moved, an
// unusual centring), which depend on their rotations and centring alone:
// those of the tabulated group with their rotations and centring (the
// translations removed); else, for a centring by thirds (R on hexagonal
// axes in the reverse setting, which gemmi does not tabulate), hexagonal
// axes; else those of the tabulated primitive group with their rotations.
// @p like is set to the tabulated group used, or nullptr.
Axes axes_of_ops(gemmi::GroupOps ops, const gemmi::SpaceGroup *&like) {
  for (gemmi::Op &op : ops.sym_ops) op.tran = {0, 0, 0};
  if ((like = gemmi::find_spacegroup_by_ops(ops))) return axes_of(*like);
  constexpr int den = gemmi::Op::DEN;
  for (const gemmi::Op::Tran &t : ops.cen_ops)
    for (const int k : t) {
      const int r = ((k % den) + den) % den;
      if (r == den / 3 || r == 2 * den / 3) return {Axes::Hexagonal, 0};
    }
  ops.cen_ops = {gemmi::Op::Tran{0, 0, 0}};
  if ((like = gemmi::find_spacegroup_by_ops(ops))) return axes_of(*like);
  return {};
}

// A monoclinic group is described in the setting its unique axis names,
// and an R group on the hexagonal or rhombohedral axes it names: in a cell
// whose angles say otherwise the operators place the images of a site
// where that crystal has none. @p what names the group, @p advice says
// what to give instead.
void require_cell_fits(const gemmi::UnitCell &c, const Axes &axes,
                       const std::string &what, const std::string &advice,
                       const std::string &source) {
  if (fits_cell(c, axes)) return;
  std::string needs;
  switch (axes.kind) {
    case Axes::Monoclinic:
      needs = std::string("is monoclinic with unique axis ") + axes.unique +
              ", which needs " +
              (axes.unique == 'a'   ? "beta = gamma"
               : axes.unique == 'b' ? "alpha = gamma"
                                    : "alpha = beta") +
              " = 90 degrees (the angles between " + axes.unique +
              " and the other axes, within 1e-3)";
      break;
    case Axes::Hexagonal:
      needs = "is on hexagonal axes, which need alpha = beta = 90 and "
              "gamma = 120 degrees (within 1e-3)";
      break;
    default:
      needs = "is on rhombohedral axes, which need alpha = beta = gamma "
              "(within 1e-3 degrees)";
  }
  using tadah::core::format_roundtrip;
  throw std::runtime_error(
      source + ": " + what + " " + needs + ", but the cell has alpha " +
      format_roundtrip(c.alpha) + ", beta " + format_roundtrip(c.beta) +
      ", gamma " + format_roundtrip(c.gamma) + "; " + advice + ".");
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

  // Errors name the file, or the text and then its data block
  const std::string where = path.empty() ? "CIF text" : "CIF file " + path;
  // One data block, one structure
  Document doc;
  Block *only = nullptr;
  try {
    doc = read_string(fileContents);
    only = &doc.sole_block();
  } catch (const std::exception &e) {
    throw std::runtime_error(where + ": " + e.what());
  }
  Block &block = *only;
  const std::string source =
      path.empty() ? "CIF text (data_" + block.name + ")" : where;

  // gemmi parses the first number tag present, which must be an integer
  // or unknown; the reader reads the number itself, where it is used, so
  // gemmi gets the block without it.
  gemmi::SmallStructure struc;
  try {
    Block without_number = block;
    for (const char *tag :
         {"_space_group_IT_number", "_symmetry_Int_Tables_number"})
      if (without_number.find_value(tag)) without_number.set_pair(tag, "?");
    struc = gemmi::make_small_structure_from_block(without_number);
  } catch (const std::exception &e) {
    throw std::runtime_error(source + ": gemmi cannot read the data block: " +
                             e.what());
  }
  // gemmi takes the first H-M and number tag present and the last Hall tag
  // present, even when its value is unknown ('?'), which then hides the
  // other tag of the pair: take the first known value of each pair, and
  // tell '0' from '?' (gemmi reads both as 0).
  const auto known_value = [&](const char *new_tag, const char *old_tag) {
    for (const char *tag : {new_tag, old_tag})
      if (const std::string *v = block.find_value(tag)) {
        std::string text = is_null(*v) ? std::string() : as_string(*v);
        trim(text);
        if (!text.empty()) return v;
      }
    return static_cast<const std::string *>(nullptr);
  };
  const std::string *hm_value =
      known_value("_space_group_name_H-M_alt", "_symmetry_space_group_name_H-M");
  const std::string *hall_value =
      known_value("_space_group_name_Hall", "_symmetry_space_group_name_Hall");
  const std::string *number_value =
      known_value("_space_group_IT_number", "_symmetry_Int_Tables_number");
  // The operator loop likewise: the first loop tag with a known value
  struc.symops.clear();
  for (const char *tag :
       {"_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"})
    if (const Column col = block.find_values(tag)) {
      bool known = false;
      for (const std::string &v : col) {
        std::string text = is_null(v) ? std::string() : as_string(v);
        trim(text);
        known = known || !text.empty();
      }
      if (!known) continue;
      for (const std::string &v : col) struc.symops.push_back(as_string(v));
      break;
    }
  struc.spacegroup_hm = hm_value ? as_string(*hm_value) : std::string();
  struc.spacegroup_hall = hall_value ? as_string(*hall_value) : std::string();
  // The number is read where it is used (no loop, no symbol).
  struc.spacegroup_number = 0;
  const bool number_given = number_value != nullptr;
  // The H-M symbol as gemmi's table lookup reads it, for every use below
  // and gemmi's consistency check. A value that normalises to nothing
  // ('(No. 14)') is kept as written, a symbol that cannot be read.
  const std::string hm_as_written = struc.spacegroup_hm;
  {
    std::string blank = hm_as_written;
    trim(blank);
    const std::string normal = normalised_hm(hm_as_written);
    struc.spacegroup_hm =
        blank.empty() ? "" : normal.empty() ? hm_as_written : normal;
  }

  // The space group, looked up here rather than with gemmi's
  // determine_and_set_spacegroup("S.H2") (which never uses the number, and
  // reaches the symbols without a loop only because it swallows the
  // std::length_error its "S." step throws on the empty list): the
  // operator loop, as gemmi does; else (no loop, or one that forms no
  // group) the Hall symbol, else the H-M symbol, each in the setting it
  // names or else origin choice 2 and hexagonal or rhombohedral axes from
  // the cell angles; else, without a loop, the number; else P 1. A symbol
  // that cannot be read is refused: the setting it means cannot be told,
  // and the next record would be a guess.
  SymmetrySource src;
  struc.spacegroup = nullptr;
  struc.setup_cell_images();
  const auto cannot_read = [&](const std::string &kind,
                               const std::string &symbol,
                               const std::string &why) {
    return std::runtime_error(
        source + ": cannot read the " + kind + " symbol '" + symbol + "'" +
        src.note + ": " + why + "; give " +
        (kind == "Hall"
             ? "a Hall symbol in standard form (e.g. '-F 4vw 2vw 3') or "
               "none (the Hall symbol is read before the H-M symbol)"
             : "it in a standard form (e.g. 'F d -3 m:2', or 'F d -3 m Z' "
               "for origin choice 2)") +
        ", or the symmetry-operator loop.");
  };
  // The Hall symbol, else the H-M symbol; false when there is neither.
  const auto from_symbols = [&]() {
    if (!struc.spacegroup_hall.empty()) {
      src.kind = SymmetrySource::Hall;
      src.record = "the Hall symbol '" + struc.spacegroup_hall + "'";
      gemmi::GroupOps ops;
      try {
        ops = gemmi::symops_from_hall(struc.spacegroup_hall.c_str());
      } catch (const std::exception &e) {
        throw cannot_read("Hall", struc.spacegroup_hall,
                          std::string("gemmi: ") + e.what());
      }
      struc.spacegroup = gemmi::find_spacegroup_by_ops(ops);
      if (struc.spacegroup) {
        struc.setup_cell_images();
      } else {  // a group in no tabulated setting: its own operators
        struc.cell.set_cell_images_from_groupops(ops);
        src.own_ops = ops;
      }
      return true;
    }
    if (!struc.spacegroup_hm.empty()) {
      src.kind = SymmetrySource::HM;
      src.record = "the H-M symbol '" + hm_as_written + "'";
      // gemmi's name lookup takes a number as a CCP4 code, which is the IT
      // number only up to 230 (1003 is P 1 1 2): a number is no H-M symbol.
      if (std::isdigit(static_cast<unsigned char>(struc.spacegroup_hm[0])))
        throw cannot_read("H-M", hm_as_written,
                          "it is a number, not an H-M symbol (the number "
                          "goes in _space_group_IT_number)");
      struc.spacegroup = gemmi::find_spacegroup_by_name(
          struc.spacegroup_hm, struc.cell.alpha, struc.cell.gamma, "2");
      if (!struc.spacegroup)
        throw cannot_read("H-M", hm_as_written,
                          "it names no space group gemmi knows");
      // gemmi prefers origin choice 2 only for the symbols of its main
      // table; one of its other names ('C c c e', the current symbol of
      // C c c a) gives origin choice 1. Its table lists origin choice 2
      // right after 1.
      const gemmi::SpaceGroup *const end =
          std::end(gemmi::spacegroup_tables::main);
      if (struc.spacegroup_hm.find(':') == std::string::npos &&
          struc.spacegroup->ext == '1' && struc.spacegroup + 1 != end &&
          struc.spacegroup[1].number == struc.spacegroup->number &&
          struc.spacegroup[1].ext == '2')
        ++struc.spacegroup;
      struc.setup_cell_images();
      return true;
    }
    return false;
  };
  bool number_has_choice = false;
  if (!struc.symops.empty()) {
    bool forms_group = false;
    try {
      const gemmi::GroupOps ops = gemmi::split_centering_vectors(
          gemmi::triplets_to_ops(struc.symops));
      struc.spacegroup = gemmi::find_spacegroup_by_ops(ops);
      if (struc.spacegroup) {
        struc.setup_cell_images();
        forms_group = true;
      } else if (ops.order() == static_cast<int>(struc.symops.size()) &&
                 gemmi::is_complete(ops)) {
        // a group in no tabulated setting: its own operators
        struc.cell.set_cell_images_from_groupops(ops);
        src.own_ops = ops;
        forms_group = true;
      }
    } catch (const std::exception &) {
      // an operator gemmi cannot parse: the loop forms no group
    }
    if (forms_group) {
      src.kind = SymmetrySource::Loop;
      src.record = "the symmetry-operator loop";
    } else {
      // gemmi's fallback; the number is not used (see the header)
      src.note = " (the operator loop forms no group)";
      from_symbols();
    }
  } else if (!from_symbols() && number_given) {
    try {
      struc.spacegroup_number = as_int(as_string(*number_value));
    } catch (const std::exception &) {
      throw std::runtime_error(
          source + ": the space-group number '" + as_string(*number_value) +
          "' is not an integer; give the number (1 to 230), the H-M symbol "
          "or the symmetry-operator loop.");
    }
    const int n = struc.spacegroup_number;
    src.kind = SymmetrySource::Number;
    src.record = "the number " + std::to_string(n);
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
    number_has_choice =
        settings_of_the_cell(*struc.spacegroup, struc.cell) > 1;
  }

  // However the group was given, a monoclinic or R one must fit the cell.
  {
    const gemmi::SpaceGroup *like = nullptr;
    const Axes axes = struc.spacegroup ? axes_of(*struc.spacegroup)
                      : src.own_ops    ? axes_of_ops(*src.own_ops, like)
                                       : Axes{};
    const std::string what =
        struc.spacegroup
            ? "the space group " + struc.spacegroup->xhm() + " (from " +
                  src.record + src.note + ")"
            : "the group of " + src.record + src.note +
                  " (in no tabulated setting; " +
                  (like ? "with the axes of " + like->xhm()
                        : std::string("R-centred on hexagonal axes")) +
                  ")";
    // An R group's axes come from the cell unless the symbol names them
    const bool axes_from_cell =
        src.kind == SymmetrySource::Number ||
        (src.kind == SymmetrySource::HM &&
         struc.spacegroup_hm.find(':') == std::string::npos);
    std::string advice;
    switch (src.kind) {
      case SymmetrySource::Loop:
        advice = "correct the operator loop or the cell (the loop is read "
                 "before the Hall and H-M symbols)";
        break;
      case SymmetrySource::Hall:
        advice = "correct the Hall symbol or the cell (the Hall symbol is "
                 "read before the H-M symbol)";
        break;
      default:
        if (axes.kind == Axes::Monoclinic)
          advice = "give the full H-M symbol of the setting the cell is in "
                   "(e.g. 'P 1 1 21/m' for unique axis c), or its "
                   "symmetry-operator loop";
        else if (axes_from_cell)
          advice = "the axes are taken from the cell angles, which fit "
                   "neither hexagonal (alpha = beta = 90, gamma = 120) nor "
                   "rhombohedral (alpha = beta = gamma) axes: correct the "
                   "cell, or give the symmetry-operator loop";
        else
          advice = "give the H-M symbol without its axes (e.g. 'R -3 m', "
                   "which takes them from the cell), or the "
                   "symmetry-operator loop";
    }
    require_cell_fits(struc.cell, axes, what, advice, source);
  }
  if (number_has_choice)
    std::cerr << "WARNING: " << source
              << ": the space group is given only by its number ("
              << struc.spacegroup_number
              << "), and the cell could be in more than one of its "
                 "settings; the unit cell is built with space group "
              << struc.spacegroup->xhm()
              << ", its standard setting (origin choice 2 where the group "
                 "has two): give the H-M symbol if the cell is in another."
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
  std::string diagnostic;
  // The number, for gemmi's check, when it was not used
  if (number_given && src.kind != SymmetrySource::Number) {
    try {
      struc.spacegroup_number = as_int(as_string(*number_value));
    } catch (const std::exception &) {
      diagnostic += "space group number ('" + as_string(*number_value) +
                    "') is not an integer\n";
    }
  }
  // gemmi's check takes an H-M symbol that is a number as a CCP4 code:
  // report it as a number instead.
  if (!struc.spacegroup_hm.empty() &&
      std::isdigit(static_cast<unsigned char>(struc.spacegroup_hm[0]))) {
    diagnostic += "H-M symbol (" + hm_as_written +
                  ") is a number, not an H-M symbol\n";
    const std::string hm = struc.spacegroup_hm;
    struc.spacegroup_hm.clear();
    diagnostic += struc.check_spacegroup();
    struc.spacegroup_hm = hm;
  } else {
    diagnostic += struc.check_spacegroup();
  }
  // gemmi compares the H-M symbol without its setting: one that names a
  // setting (':1', ':2', ':H', ':R', ICSD's ' S' or ' Z') other than the
  // one the cell is built with, in the same group, is reported too.
  if (struc.spacegroup && src.kind != SymmetrySource::HM &&
      struc.spacegroup_hm.find(':') != std::string::npos) {
    const gemmi::SpaceGroup *named = gemmi::find_spacegroup_by_name(
        struc.spacegroup_hm, struc.cell.alpha, struc.cell.gamma, "2");
    if (named && named != struc.spacegroup &&
        std::strcmp(named->hm, struc.spacegroup->hm) == 0)
      diagnostic += "H-M symbol (" + hm_as_written + ") names " +
                    named->xhm() + ", not " + struc.spacegroup->xhm() + "\n";
  }
  if (!diagnostic.empty()) {
    std::cerr << "WARNING: " << source
              << ": the symmetry record is inconsistent (\""
              << one_line(diagnostic) << "\"); the unit cell is built with "
              << symmetry_used(struc, src) << "." << std::endl;
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
