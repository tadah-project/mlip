#include <tadah/core/periodic_table.h>
#include <tadah/core/utils.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>
#include <tadah/mlip/dataset_readers/castep_species.h>
#include <tadah/mlip/dataset_readers/missing_stress.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

/* Electron masses per atomic mass unit (CODATA 2018): CASTEP's .md
 * velocities and kinetic energy are in atomic units. */
constexpr double kElectronMassesPerU = 1822.888486209;

/* Kinetic energies that differ by more than this (relative) come from
 * other masses than the periodic table's: a custom SPECIES_MASS (D, 6Li or
 * pure 28Si differ by 4e-3 or more). Mass tables of different codes differ
 * by less (CASTEP's default Si 28.0855 u against 28.0850 u is 1.8e-5). */
constexpr double kKineticEnergyTolerance = 1e-3;

using NumberedLine = CastepMDReader::FrameLine;

/* |det| of the lattice rows: the cell volume. */
double cell_volume(const tadah::core::Matrix3d &c) {
  return std::abs(c(0, 0) * (c(1, 1) * c(2, 2) - c(1, 2) * c(2, 1)) -
                  c(0, 1) * (c(1, 0) * c(2, 2) - c(1, 2) * c(2, 0)) +
                  c(0, 2) * (c(1, 0) * c(2, 1) - c(1, 1) * c(2, 0)));
}

/* The tag of a '<-- X' line ("E", "h", "S", "R", "V", "F", "c", ...), or ""
 * for an untagged line. */
std::string line_tag(const std::string &text) {
  const auto p = text.rfind("<--");
  if (p == std::string::npos) return "";
  std::istringstream iss(text.substr(p + 3));
  std::string tag;
  iss >> tag;
  return tag;
}

/* The fields before the tag. */
std::vector<std::string> line_fields(const std::string &text) {
  const auto p = text.rfind("<--");
  std::istringstream iss(text.substr(0, p));
  std::vector<std::string> out;
  std::string f;
  while (iss >> f) out.push_back(f);
  return out;
}

/* A problem in one frame: the line it is on and what it is. `incomplete`:
 * lines are missing, as when the file ends inside the frame. */
struct FrameError {
  size_t line;
  std::string what;
  bool incomplete = false;
};

/* `text` as a number, all of it; throws FrameError otherwise. */
double frame_number(const std::string &text, size_t line,
                    const std::string &what) {
  char *end = nullptr;
  const double v = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || !std::isfinite(v))
    throw FrameError{line, "'" + text + "' is not a number (" + what + ")"};
  return v;
}

/* The three numbers of an h/S line, or of the vector of an R/V/F line
 * (after its species label and index). */
void frame_vector(const NumberedLine &l, size_t first, const std::string &what,
                  double out[3]) {
  const auto f = line_fields(l.text);
  if (f.size() != first + 3)
    throw FrameError{l.number, "expected " + std::to_string(first + 3) +
                                   " fields before '<--', found " +
                                   std::to_string(f.size()) + " (" + what +
                                   ")"};
  for (size_t k = 0; k < 3; ++k) out[k] = frame_number(f[first + k], l.number, what);
}

} // namespace

CastepMDReader::CastepMDReader(StructureDB& db)
: DatasetReader(db) {}

CastepMDReader::CastepMDReader(StructureDB& db, const std::string& filename)
: DatasetReader(db, filename) {
  read_data(filename);
}

void CastepMDReader::read_data(const std::string& filename) {
  filename_ = filename;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open the file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    raw_data_ += line + "\n";
  }
  file.close();
}

bool CastepMDReader::ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
  str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string CastepMDReader::get_first_label(std::string &time) {
  return "Filename: "+filename_+ " | Units: (eV, Angstrom) | Time: "
  + std::to_string(std::stod(time)*t_conv) + " ps";
}
std::string CastepMDReader::get_label(std::string &time) {
  return "Time: " + std::to_string(std::stod(time)*t_conv) + " ps";
}

/* A frame: its first line holds the time (.md, untagged) or the step
 * (.geom, tagged 'c'), then the tagged lines: '<-- E' (energy, Hamiltonian
 * and, in .md, the kinetic energy), '<-- T' (k_B T), three '<-- h' (the
 * lattice vectors, rows), none or three '<-- S' (the stress, Ha/Bohr^3,
 * tension positive), one '<-- R' per atom, in .md one '<-- V' per atom in
 * the same order, and one '<-- F' per atom. Atomic units throughout. */
Structure CastepMDReader::parse_frame(const std::vector<FrameLine> &frame,
                                      size_t frame_index) {
  const NumberedLine &first = frame.front();
  const std::string first_tag = line_tag(first.text);
  if (!first_tag.empty() && first_tag != "c")
    throw FrameError{first.number,
                     "the frame does not start with its time or step line"};
  const auto first_fields = line_fields(first.text);
  if (first_fields.empty())
    throw FrameError{first.number, "the time or step line is empty"};
  std::string time = first_fields[0];
  frame_number(time, first.number, has_velocities() ? "time" : "step");

  const NumberedLine *e_line = nullptr;
  const NumberedLine *t_line = nullptr;
  std::vector<const NumberedLine *> h, st, r, v, f;
  for (size_t i = 1; i < frame.size(); ++i) {
    const NumberedLine &l = frame[i];
    const std::string tag = line_tag(l.text);
    if (tag == "E") {
      if (e_line) throw FrameError{l.number, "a second '<-- E' line"};
      e_line = &l;
    } else if (tag == "T") {
      if (t_line) throw FrameError{l.number, "a second '<-- T' line"};
      t_line = &l;
    } else if (tag == "h") {
      h.push_back(&l);
    } else if (tag == "S") {
      st.push_back(&l);
    } else if (tag == "R") {
      r.push_back(&l);
    } else if (tag == "V") {
      v.push_back(&l);
    } else if (tag == "F") {
      f.push_back(&l);
    } else if (tag.empty()) {
      throw FrameError{l.number, "an untagged line inside the frame"};
    }
    // Other tags (P, hv, ...) carry nothing Tadah stores.
  }
  const size_t start = first.number;
  auto count_error = [&](const std::string &what, size_t want, size_t found) {
    return FrameError{start,
                      "expected " + std::to_string(want) + " '<-- " + what +
                          "' line(s), found " + std::to_string(found),
                      true};
  };
  if (!e_line) throw count_error("E", 1, 0);
  if (h.size() != 3) throw count_error("h", 3, h.size());
  if (!st.empty() && st.size() != 3) throw count_error("S", 3, st.size());
  if (r.empty()) throw FrameError{start, "no '<-- R' (position) lines", true};
  if (f.size() != r.size()) throw count_error("F", r.size(), f.size());
  const bool velocities = has_velocities();
  if (velocities && v.size() != r.size()) throw count_error("V", r.size(), v.size());

  Structure s;
  const auto e_fields = line_fields(e_line->text);
  if (e_fields.size() < (velocities ? 3u : 1u))
    throw FrameError{e_line->number, "too few values on the '<-- E' line"};
  s.energy = frame_number(e_fields[0], e_line->number, "energy") * e_conv;
  double temperature = 0;  // k_B T, hartree
  if (t_line) {
    const auto t_fields = line_fields(t_line->text);
    if (t_fields.empty())
      throw FrameError{t_line->number, "no value on the '<-- T' line"};
    temperature = frame_number(t_fields[0], t_line->number, "temperature");
  }

  // Cell rows (Bohr until the end).
  tadah::core::Matrix3d cell_bohr;
  for (size_t k = 0; k < 3; ++k) {
    double row[3];
    frame_vector(*h[k], 0, "lattice vector", row);
    for (size_t j = 0; j < 3; ++j) cell_bohr(k, j) = row[j];
  }

  // Atoms: the species label and index of each V and F line must be those
  // of the R line in the same place.
  std::vector<std::string> labels(r.size());
  for (size_t a = 0; a < r.size(); ++a) {
    const auto fields = line_fields(r[a]->text);
    if (fields.size() != 5)
      throw FrameError{r[a]->number, "expected species, index and three "
                                     "coordinates before '<-- R'"};
    labels[a] = fields[0] + " " + fields[1];
    double pos[3];
    frame_vector(*r[a], 2, "position", pos);
    tadah::core::Element element;
    try {
      element = castep_species_element(fields[0]);
    } catch (const std::exception &ex) {
      throw FrameError{r[a]->number, ex.what()};
    }
    s.add_atom(Atom(element, pos[0] * d_conv, pos[1] * d_conv,
                    pos[2] * d_conv, 0, 0, 0));
  }
  auto same_atom = [&](const NumberedLine &l, size_t a, const char *what) {
    const auto fields = line_fields(l.text);
    if (fields.size() < 2 || fields[0] + " " + fields[1] != labels[a])
      throw FrameError{l.number, std::string("the ") + what +
                                     " line does not follow the order of the "
                                     "'<-- R' lines (" + labels[a] + ")"};
  };
  for (size_t a = 0; a < f.size(); ++a) {
    same_atom(*f[a], a, "'<-- F'");
    double force[3];
    frame_vector(*f[a], 2, "force", force);
    for (size_t k = 0; k < 3; ++k) s.atoms[a].force[k] = force[k] * f_conv;
  }

  if (!st.empty()) {
    tadah::core::Matrix3d stress;  // Ha/Bohr^3
    for (size_t k = 0; k < 3; ++k) {
      double row[3];
      frame_vector(*st[k], 0, "stress", row);
      for (size_t j = 0; j < 3; ++j) stress(k, j) = row[j];
    }
    if (velocities) {
      // The .md stress includes the kinetic term -(1/V) sum m v (x) v; the
      // model predicts only the configurational part (and LAMMPS adds the
      // kinetic part itself), so it is removed exactly. Masses: the
      // periodic table's; K is scaled so that its trace is twice the
      // file's kinetic energy, which is what CASTEP added.
      tadah::core::Matrix3d K;
      std::vector<std::string> species;
      for (size_t a = 0; a < v.size(); ++a) {
        same_atom(*v[a], a, "'<-- V'");
        double vel[3];
        frame_vector(*v[a], 2, "velocity", vel);
        const double m =
            tadah::core::PeriodicTable::get_mass(s.atoms[a]) * kElectronMassesPerU;
        for (size_t i = 0; i < 3; ++i)
          for (size_t j = 0; j < 3; ++j) K(i, j) += m * vel[i] * vel[j];
        if (std::find(species.begin(), species.end(), s.atoms[a].symbol) ==
            species.end())
          species.push_back(s.atoms[a].symbol);
      }
      const double ke_file =
          frame_number(e_fields[2], e_line->number, "kinetic energy");
      const double ke_masses = 0.5 * (K(0, 0) + K(1, 1) + K(2, 2));
      if (ke_file != 0 || ke_masses != 0) {
        const double ratio = ke_masses != 0 ? ke_file / ke_masses : 0;
        if (!(std::abs(ratio - 1) <= kKineticEnergyTolerance)) {
          std::string names;
          for (const auto &sp : species) names += (names.empty() ? "" : ", ") + sp;
          std::ostringstream os;
          os << "the kinetic energy of the velocities with the periodic-table "
                "masses of " << names << " is " << ke_masses
             << " Ha, the file's '<-- E' kinetic energy " << ke_file
             << " Ha: the run used other masses (SPECIES_MASS), which a .md "
                "file does not record, so its kinetic stress cannot be removed";
          throw FrameError{e_line->number, os.str()};
        }
        for (size_t i = 0; i < 3; ++i)
          for (size_t j = 0; j < 3; ++j) K(i, j) *= ratio;
      }
      const double V = cell_volume(cell_bohr);
      for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j) stress(i, j) += K(i, j) / V;
    }
    s.stress = stress;
    s.stress *= s_conv;
  } else {
    // No stress tensor: zero stress and stress weight 0, so that a STRESS
    // fit ignores it.
    s.sweight = 0.0;
  }

  s.cell = cell_bohr;
  s.cell *= d_conv;
  s.T = temperature / k_b;
  s.label = frame_index == 0 ? get_first_label(time) : get_label(time);
  return s;
}

void CastepMDReader::parse_data() {
  std::vector<NumberedLine> lines;
  {
    std::istringstream stream(raw_data_);
    std::string text;
    size_t number = 0;
    while (std::getline(stream, text)) lines.push_back({++number, text});
  }
  // Frames follow the header ("BEGIN header" ... "END header").
  size_t begin = 0;
  for (size_t k = 0; k < lines.size(); ++k) {
    if (lines[k].text.find("END header") != std::string::npos) {
      begin = k + 1;
      break;
    }
  }
  // A frame is a run of non-blank lines. The last one is "open" when the
  // file ends inside it, as when a run is stopped while writing it.
  std::vector<std::vector<NumberedLine>> frames;
  std::vector<NumberedLine> current;
  for (size_t k = begin; k < lines.size(); ++k) {
    if (tadah::core::is_blank_line(lines[k].text)) {
      if (!current.empty()) frames.push_back(std::move(current));
      current.clear();
    } else {
      current.push_back(lines[k]);
    }
  }
  const bool last_open = !current.empty();
  if (last_open) frames.push_back(std::move(current));

  size_t stored = 0, without_stress = 0;
  for (size_t i = 0; i < frames.size(); ++i) {
    try {
      const Structure st = parse_frame(frames[i], i);
      stdb.add(st);
      ++stored;
      if (st.sweight == 0.0) ++without_stress;
    } catch (const FrameError &e) {
      // The file ends inside its last frame: lines are missing, or the last
      // line was cut. Any other problem is an error wherever it is.
      const bool truncated =
          last_open && i + 1 == frames.size() &&
          (e.incomplete || e.line == frames[i].back().number);
      if (truncated) {
        std::cerr << "WARNING: " << filename_ << ": the last frame (" << i + 1
                  << ", from line " << frames[i].front().number
                  << ") is incomplete and was skipped (line " << e.line
                  << ": " << e.what << "); " << i << " structure(s) read"
                  << std::endl;
        break;
      }
      throw std::runtime_error(filename_ + ": frame " + std::to_string(i + 1) +
                               ", line " + std::to_string(e.line) + ": " +
                               e.what);
    }
  }
  log_missing_stress("CASTEP " + filename_, without_stress, stored);
}

void CastepMDReader::print_summary() const {
  std::cout << get_summary();
}

std::string CastepMDReader::get_summary() const {
  return stdb.summary();
}
} // namespace mlip
} // namespace tadah
