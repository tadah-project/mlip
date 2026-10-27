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
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
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

// The same, for the structure's label: the tabulated setting ("F d -3 m:2",
// "R -3 m:H"), the operators of a group in no tabulated setting ("2
// operators of the symmetry-operator loop"), or P 1.
std::string group_label(const gemmi::SmallStructure &struc,
                        const SymmetrySource &src) {
  if (struc.spacegroup) return struc.spacegroup->xhm();
  if (src.own_ops)
    return std::to_string(src.own_ops->order()) + " operators of " +
           src.record;
  return "P 1";
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
// spglib) or in parentheses ('P 4(2)/n m c'), a trailing ' (No. 14)', the
// notes '(origin choice 2)', '(origin at -1)' and '(hexagonal axes)', and
// ICSD's origin-choice suffix ' S' or ' Z' (origin choice 1 or 2), which
// gemmi reads as ':1' or ':2'.
std::string normalised_hm(std::string hm) {
  static const std::regex number_note(R"(\s*\(\s*[Nn][Oo]\.?\s*\d+\s*\)$)");
  static const std::regex subscript(R"((\d)\((\d)\))");
  // The notes of the CIF dictionary's examples: the origin choice, the
  // origin at the centre of symmetry (origin choice 2), the axes
  static const std::regex origin_choice(
      R"(\s*\(\s*origin\s+choice\s+([12])\s*\)$)", std::regex::icase);
  static const std::regex origin_at_centre(
      R"(\s*\(\s*origin\s+at\s+-1\s*\)$)", std::regex::icase);
  static const std::regex hexagonal_axes(
      R"(\s*\(\s*hexagonal\s+axes\s*\)$)", std::regex::icase);
  static const std::regex rhombohedral_axes(
      R"(\s*\(\s*rhombohedral\s+axes\s*\)$)", std::regex::icase);
  trim(hm);
  hm = std::regex_replace(hm, number_note, "");
  hm = std::regex_replace(hm, origin_choice, ":$1");
  hm = std::regex_replace(hm, origin_at_centre, ":2");
  hm = std::regex_replace(hm, hexagonal_axes, ":H");
  hm = std::regex_replace(hm, rhombohedral_axes, ":R");
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
    throw std::logic_error("internal error: monoclinic space group " + sg.xhm() +
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

// The axes a group's rotations fix, which the cell angles must fit: a
// monoclinic group's unique axis (the angles between it and the other two
// axes are 90 degrees; the third is free), hexagonal axes (alpha = beta =
// 90, gamma = 120: any 3-fold rotation about c) or an R group's
// rhombohedral axes (alpha = beta = gamma). Other groups are not checked
// (Free).
struct Axes {
  enum Kind { Free, Monoclinic, Hexagonal, Rhombohedral } kind = Free;
  char unique = 0;  // Monoclinic: 'a', 'b' or 'c'
};

// Whether @p ops hold a 3-fold rotation (proper or not) about c on
// hexagonal axes, -y,x-y,z or -x+y,-x,z: the rotation that needs alpha =
// beta = 90 and gamma = 120 (every trigonal and hexagonal group on such
// axes, R or P, has one; a centring or other translation fixes nothing).
bool has_hexagonal_threefold(const gemmi::GroupOps &ops) {
  constexpr int d = gemmi::Op::DEN;
  using Rot = gemmi::Op::Rot;
  const Rot r1 = {{{0, -d, 0}, {d, -d, 0}, {0, 0, d}}};
  const Rot r2 = {{{-d, d, 0}, {-d, 0, 0}, {0, 0, d}}};
  for (const gemmi::Op &op : ops.sym_ops) {
    const Rot neg = op.negated_rot();
    if (op.rot == r1 || op.rot == r2 || neg == r1 || neg == r2) return true;
  }
  return false;
}

Axes axes_of(const gemmi::SpaceGroup &sg) {
  if (const char axis = monoclinic_axis(sg)) return {Axes::Monoclinic, axis};
  if (sg.hm[0] == 'R' && sg.ext == 'R') return {Axes::Rhombohedral, 0};
  if (has_hexagonal_threefold(sg.operations())) return {Axes::Hexagonal, 0};
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

// The setting a (normalised) H-M symbol names in cell @p c, or nullptr:
// gemmi's lookup with origin choice 2 and hexagonal or rhombohedral axes
// from the cell angles where the symbol names none. gemmi prefers origin
// choice 2 only for the symbols of its main table; one of its other names
// ('C c c e', the current symbol of C c c a) gives origin choice 1. Its
// table lists origin choice 2 right after 1.
// The short H-M symbol of a full one (ITA): the 'n/x' of a symmetry
// direction is reduced to 'x', in every direction for an orthorhombic or
// cubic group ('P 2/n 2/n 2/n' is 'P n n n', 'I 4/m -3 2/m' 'I m -3 m'),
// in all but the first for a tetragonal, trigonal or hexagonal one
// ('P 63/m 2/m 2/c' is 'P 63/m m c', 'R -3 2/m' 'R -3 m'). A monoclinic
// full symbol ('P 1 21/c 1') is gemmi's own; it gives "".
std::string short_hm(const std::string &full) {
  std::istringstream in(full);
  std::vector<std::string> t;
  for (std::string w; in >> w;) t.push_back(w);
  if (t.size() < 3) return "";
  const auto starts = [](const std::string &w, char d) {
    return w[0] == d || (w[0] == '-' && w.size() > 1 && w[1] == d);
  };
  // Cubic: the 3 in the second direction (m-3: 'P 21/a -3'; m-3m:
  // 'I 4/m -3 2/m')
  const bool cubic =
      (t.size() == 3 || t.size() == 4) && (t[2] == "-3" || t[2] == "3");
  const bool principal_axis =
      starts(t[1], '4') || starts(t[1], '6') || starts(t[1], '3');
  const bool orthorhombic =
      t.size() == 4 && t[1] != "1" && t[2] != "1" && t[3] != "1";
  if (!cubic && !principal_axis && !orthorhombic) return "";
  std::string out = t[0];
  for (std::size_t i = 1; i < t.size(); ++i) {
    std::string w = t[i];
    const std::size_t slash = w.find('/');
    if (slash != std::string::npos && (cubic || !(principal_axis && i == 1)))
      w = w.substr(slash + 1);
    out += " " + w;
  }
  return out;
}

const gemmi::SpaceGroup *lookup_hm(const std::string &hm,
                                   const gemmi::UnitCell &c) {
  const auto find = [&](const std::string &name) {
    return gemmi::find_spacegroup_by_name(name, c.alpha, c.gamma, "2");
  };
  const std::size_t colon = hm.find(':');
  const std::string symbol = hm.substr(0, colon);
  const std::string setting =
      colon == std::string::npos ? "" : hm.substr(colon);
  const gemmi::SpaceGroup *sg = find(hm);
  // A full symbol, by its short one
  if (!sg) {
    const std::string short_symbol = short_hm(symbol);
    if (!short_symbol.empty()) sg = find(short_symbol + setting);
  }
  // An origin note on a group with one origin ('P 2/m 2/m 2/m (origin at
  // -1)'): the origin is that group's
  if (!sg && (setting == ":1" || setting == ":2")) {
    const std::string short_symbol = short_hm(symbol);
    const gemmi::SpaceGroup *one =
        find(short_symbol.empty() ? symbol : short_symbol);
    if (one && one->ext == 0) sg = one;
  }
  const gemmi::SpaceGroup *const end =
      std::end(gemmi::spacegroup_tables::main);
  if (sg && hm.find(':') == std::string::npos && sg->ext == '1' &&
      sg + 1 != end && sg[1].number == sg->number && sg[1].ext == '2')
    ++sg;
  return sg;
}

// Whether @p ops is closed, translations included: the product of any two
// operators (wrapped into the cell) is one of them. gemmi's is_complete()
// looks at the rotations alone.
bool closed(const gemmi::GroupOps &ops) {
  const std::vector<gemmi::Op> all = ops.all_ops_sorted();
  for (const gemmi::Op &a : all)
    for (const gemmi::Op &b : all) {
      gemmi::Op ab = a * b;
      ab.wrap();
      if (!std::binary_search(all.begin(), all.end(), ab)) return false;
    }
  return true;
}

// Whether the operators listed in a loop are, wrapped into the cell, those
// of @p ops: gemmi splits a loop into rotations and centring vectors and
// drops a listed operator that shares a rotation with another but no pure
// centring shift, so its set can be closed when the listed one is not.
bool same_operators(const std::vector<std::string> &listed,
                    const gemmi::GroupOps &ops) {
  std::vector<gemmi::Op> mine = gemmi::triplets_to_ops(listed);
  for (gemmi::Op &op : mine) op.wrap();
  std::sort(mine.begin(), mine.end());
  mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
  return mine == ops.all_ops_sorted();
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
// unusual centring), which depend on their rotations alone: those of the
// tabulated group with their rotations and centring (the translations
// removed); else hexagonal axes for a 3-fold rotation about c (R in the
// reverse setting, which gemmi does not tabulate); else those of the
// tabulated primitive group with their rotations. @p like is set to the
// tabulated group used, or nullptr.
Axes axes_of_ops(gemmi::GroupOps ops, const gemmi::SpaceGroup *&like) {
  for (gemmi::Op &op : ops.sym_ops) op.tran = {0, 0, 0};
  if ((like = gemmi::find_spacegroup_by_ops(ops))) return axes_of(*like);
  if (has_hexagonal_threefold(ops)) return {Axes::Hexagonal, 0};
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
    throw std::runtime_error("the structure file " + path + " cannot be opened");
  }

  std::ostringstream buffer;
  buffer << ifs.rdbuf();
  parseCif(buffer.str(), "CIF file " + path, false, "CIF file: " + path);
}

Structure CifReader::getStructure() const { return m_structure; }

void CifReader::parseCifContents(const std::string &fileContents) {
  parseCif(fileContents, "CIF text", true, "");
}

void CifReader::parseCifContents(const std::string &fileContents,
                                 const std::string &origin) {
  if (origin.empty()) {
    parseCifContents(fileContents);
    return;
  }
  parseCif(fileContents, origin, false, "");
}

void CifReader::parseCif(const std::string &fileContents,
                         const std::string &where, bool name_block,
                         const std::string &label_origin) {
  using namespace gemmi::cif;

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
  // Errors and WARNINGs name the file or the origin, or the text and its
  // data block
  const std::string source =
      name_block ? where + " (data_" + block.name + ")" : where;

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
  // The cell: gemmi sets it only when all six tags are given, and keeps a
  // 1 A cube otherwise. A length must be given; an angle left out (or
  // '.') is 90 degrees, the CIF dictionary's default; an unknown one ('?')
  // is refused.
  {
    double v[6];
    const char *tags[6] = {"_cell_length_a",    "_cell_length_b",
                           "_cell_length_c",    "_cell_angle_alpha",
                           "_cell_angle_beta",  "_cell_angle_gamma"};
    for (int k = 0; k < 6; ++k) {
      const std::string *value = block.find_value(tags[k]);
      const bool length = k < 3;
      if (!value || is_null(*value)) {
        if (!length && (!value || *value == "."))
          v[k] = 90.0;
        else
          throw std::runtime_error(
              source + ": " + tags[k] + " is " +
              (value ? "unknown ('" + *value + "')" : std::string("missing")) +
              "; a CIF gives the three cell lengths, and the angles that "
              "are not 90 degrees.");
        continue;
      }
      v[k] = as_number(*value);
      if (!(v[k] > 0.0) || (!length && !(v[k] < 180.0)))
        throw std::runtime_error(source + ": " + tags[k] + " is '" +
                                 *value + "', which is no " +
                                 (length ? "length" : "angle") + ".");
    }
    struc.cell.set(v[0], v[1], v[2], v[3], v[4], v[5]);
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
        src.note + ": " + why + "; the reader takes " +
        (kind == "Hall"
             ? "a Hall symbol in standard form (such as '-F 4vw 2vw 3') or "
               "none (the Hall symbol is read before the H-M symbol)"
             : "the symbol in a standard form (such as 'F d -3 m:2', or "
               "'F d -3 m Z' for origin choice 2)") +
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
      struc.spacegroup = lookup_hm(struc.spacegroup_hm, struc.cell);
      if (!struc.spacegroup)
        throw cannot_read("H-M", hm_as_written,
                          "it names no space group gemmi knows");
      struc.setup_cell_images();
      return true;
    }
    return false;
  };
  // The number, used when there is no symbol: its standard setting
  bool number_has_choice = false;
  const auto from_number = [&]() {
    try {
      struc.spacegroup_number = as_int(as_string(*number_value));
    } catch (const std::exception &) {
      throw std::runtime_error(
          source + ": the space-group number '" + as_string(*number_value) +
          "' is not an integer; the reader takes the number (1 to 230), the "
          "H-M symbol or the symmetry-operator loop.");
    }
    const int n = struc.spacegroup_number;
    src.kind = SymmetrySource::Number;
    src.record = "the number " + std::to_string(n);
    if (n < 1 || n > 230)
      throw std::runtime_error(
          source + ": the space-group number " + std::to_string(n) +
          " is not one of the 230 space groups (1 to 230); the reader takes "
          "the H-M symbol or the symmetry-operator loop.");
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
  };
  bool loop_forms_no_group = false;
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
                 gemmi::is_complete(ops) && closed(ops) &&
                 same_operators(struc.symops, ops)) {
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
      // The symbols, then the number, as without a loop
      loop_forms_no_group = true;
      src.note = " (the operator loop forms no group)";
      if (!from_symbols() && number_given) from_number();
    }
  } else if (!from_symbols() && number_given) {
    from_number();
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
                        : std::string("a 3-fold rotation about c")) +
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
      default: {
        // Another setting can fit the cell only if the cell fits its axes
        const gemmi::UnitCell &c = struc.cell;
        const bool monoclinic_fits =
            fits_cell(c, {Axes::Monoclinic, 'a'}) ||
            fits_cell(c, {Axes::Monoclinic, 'b'}) ||
            fits_cell(c, {Axes::Monoclinic, 'c'});
        const bool other_r_fits =
            fits_cell(c, {axes.kind == Axes::Hexagonal ? Axes::Rhombohedral
                                                       : Axes::Hexagonal,
                          0});
        if (axes.kind == Axes::Monoclinic)
          advice = monoclinic_fits
                       ? "the reader takes the full H-M symbol of the setting "
                         "the cell is in (such as 'P 1 1 21/m' for unique "
                         "axis c), or its symmetry-operator loop"
                       : "the cell angles fit no monoclinic setting (two of "
                         "alpha, beta and gamma are 90), so the cell does "
                         "not match the symbol";
        else if (struc.spacegroup == nullptr ||
                 struc.spacegroup->hm[0] != 'R')
          advice = "the cell angles do not fit hexagonal axes (alpha = beta "
                   "= 90, gamma = 120), so the cell does not match the "
                   "symbol";
        else if (axes_from_cell || !other_r_fits)
          advice = std::string(axes_from_cell
                                   ? "the axes are taken from the cell "
                                     "angles, which"
                                   : "the cell angles") +
                   " fit neither hexagonal (alpha = beta = 90, gamma = 120) "
                   "nor rhombohedral (alpha = beta = gamma) axes, so the cell "
                   "does not match the symbol";
        else
          advice = "the reader takes the H-M symbol without its axes (such "
                   "as 'R -3 m', which takes them from the cell), or the "
                   "symmetry-operator loop";
      }
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
                 "has two); for a cell in another setting the reader takes "
                 "the H-M symbol."
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
             "(disorder or a vacancy); the reader takes an ordered structure, "
             "one atom per site.";
      throw std::runtime_error(msg.str());
    }
    if (occ > 1.0 + 1e-6) {
      std::ostringstream msg;
      msg << source << ": site " << label << " has occupancy " << text
          << ", above 1, which is not a valid occupancy.";
      throw std::runtime_error(msg.str());
    }
  }

  // Every site is of an element Tadah knows (H to Og). gemmi also reads
  // D (deuterium), which Tadah has not, and takes a type symbol (or, without
  // the type-symbol column, a label) it cannot read as X.
  for (const gemmi::SmallStructure::Site &site : struc.sites) {
    if (site.element == gemmi::El::X)
      throw std::runtime_error(source + ": site " + site.label + ": '" +
                               site.type_symbol +
                               "' is not an element symbol.");
    const std::string symbol = site.element.name();
    try {
      tadah::core::PeriodicTable::find_by_symbol(symbol);
    } catch (const std::runtime_error &) {
      throw std::runtime_error(source + ": site " + site.label +
                               " has element '" + symbol +
                               "', which Tadah does not know: its elements "
                               "are H to Og.");
    }
  }

  // A record that contradicts itself (or names nothing gemmi knows) is
  // reported; the structure is built with what gemmi found first.
  std::string diagnostic;
  if (loop_forms_no_group)
    diagnostic += "the operator loop forms no group (not closed under "
                  "products, translations included)\n";
  // The number, for gemmi's check, when it was not used
  if (number_given && src.kind != SymmetrySource::Number) {
    try {
      struc.spacegroup_number = as_int(as_string(*number_value));
    } catch (const std::exception &) {
      diagnostic += "space group number ('" + as_string(*number_value) +
                    "') is not an integer\n";
    }
  }
  // gemmi's check, but for the H-M symbol, which it compares by its
  // string without the setting (and takes as a CCP4 code when it is a
  // number): the reader compares it by the operations it names.
  {
    const std::string hm = struc.spacegroup_hm;
    struc.spacegroup_hm.clear();
    diagnostic += struc.check_spacegroup();
    struc.spacegroup_hm = hm;
  }
  if (!struc.spacegroup_hm.empty() && src.kind != SymmetrySource::HM) {
    const gemmi::SpaceGroup *used = struc.spacegroup;
    const gemmi::SpaceGroup *named =
        std::isdigit(static_cast<unsigned char>(struc.spacegroup_hm[0]))
            ? nullptr
            : lookup_hm(struc.spacegroup_hm, struc.cell);
    if (std::isdigit(static_cast<unsigned char>(struc.spacegroup_hm[0])))
      diagnostic += "H-M symbol (" + hm_as_written +
                    ") is a number, not an H-M symbol\n";
    else if (!named)
      diagnostic +=
          "H-M symbol (" + hm_as_written + ") not found in the table\n";
    else if (!used)
      diagnostic += "space group from H-M symbol (" + hm_as_written +
                    ") is " + named->xhm() +
                    ", which differs from the operators used\n";
    else if (named != used &&
             !named->operations().is_same_as(used->operations())) {
      // A symbol that names a setting (':1', ':2', ':H', ':R', ICSD's
      // suffixes) must name the one used; one that names none, its group.
      if (struc.spacegroup_hm.find(':') != std::string::npos)
        diagnostic += "H-M symbol (" + hm_as_written + ") names " +
                      named->xhm() + ", not " + used->xhm() + "\n";
      else {
        // A name without a setting agrees with any setting of that name
        bool agrees = false;
        for (const gemmi::SpaceGroup &t : gemmi::spacegroup_tables::main)
          agrees = agrees || (std::strcmp(t.hm, named->hm) == 0 &&
                              t.operations().is_same_as(used->operations()));
        if (!agrees)
          diagnostic += "space group from H-M symbol (" + hm_as_written +
                        ") is " + named->hm + ", which differs from " +
                        used->xhm() + "\n";
      }
    }
  }
  if (!diagnostic.empty()) {
    std::cerr << "WARNING: " << source
              << ": the symmetry record is inconsistent (\""
              << one_line(diagnostic) << "\"); the unit cell is built with "
              << symmetry_used(struc, src) << "." << std::endl;
  }

  auto sites = struc.get_all_unit_cell_sites();

  // The whole structure is built here and replaces the previous one only
  // when it is complete, so a refused CIF leaves it as it was. The label
  // names the space group the unit cell is built with, as the WARNINGs do
  // (the H-M text as written is in the WARNINGs, where it differs).
  Structure out;
  out.label = (label_origin.empty() ? source : label_origin) +
              " | Symmetry: " + group_label(struc, src) +
              " | Volume: " + std::to_string(struc.cell.volume);

  out.atoms.reserve(sites.size());
  for (const auto &site : sites) {
    auto p = struc.cell.orthogonalize(site.fract);
    auto label = site.element.name();
    tadah::core::Element e = tadah::core::PeriodicTable().find_by_symbol(label);
    Atom a(e, p.x, p.y, p.z, 0, 0, 0);
    out.atoms.push_back(a);
  }
  // gemmi's orthogonalisation matrix maps fractional to Cartesian
  // coordinates (r = orth * f), so its COLUMNS are the lattice vectors.
  // Structure::cell stores lattice vector a_i as row i: transpose.
  auto mat = struc.cell.orth.mat;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      out.cell(i, j) = mat.a[j][i];
    }
  }
  m_structure = std::move(out);
}
} // namespace mlip
} // namespace tadah
