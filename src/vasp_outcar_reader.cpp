#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/dataset_readers/missing_stress.h>
#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/utils.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace tadah {
namespace mlip {

VaspOutcarReader::VaspOutcarReader(StructureDB& db) : DatasetReader(db) {}

VaspOutcarReader::VaspOutcarReader(StructureDB& db, const std::string& filename) 
: DatasetReader(db, filename), filename_(filename) {
  read_data(filename);
}

void VaspOutcarReader::read_data(const std::string& filename) {
  filename_ = filename;
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("VASP OUTCAR " + filename + ": cannot open the file");
  }

  std::string line;
  while (std::getline(file, line)) {
    raw_data_ += line + "\n";
  }

  file.close();
}

void VaspOutcarReader::parse_data() {
  // order in OUTCAR is: VRHFIN, NIONS,...
  std::istringstream stream(raw_data_);
  std::string line;
  std::vector<std::string> atom_types;
  std::vector<size_t> natom_types;
  std::vector<tadah::core::Element> elements;  // of atom_types

  size_t natoms = 0;
  bool have_natoms = false;
  size_t line_no = 0;  // the line `line` holds
  auto next_line = [&]() -> bool {
    if (!std::getline(stream, line)) return false;
    ++line_no;
    return true;
  };
  // Errors name the file and the line; a block the file ends inside ends
  // the read with one warning (a run stopped while writing it).
  auto refuse = [&](const std::string &what) {
    throw std::runtime_error("VASP OUTCAR " + filename_ + ": line " +
                             std::to_string(line_no) + ": " + what);
  };
  auto quoted = [&]() { return "'" + tadah::core::trim(line) + "'"; };
  // A count: the whole token, digits, at most 10^12 (no OUTCAR has more
  // ions; an absurd count is not a count).
  auto count_of = [](const std::string &tok, size_t &out) {
    if (tok.empty() || tok.size() > 13 || !core::is_integer_literal(tok) ||
        tok[0] == '-')
      return false;
    const unsigned long long v = std::stoull(tok);
    if (v > 1000000000000ULL) return false;
    out = static_cast<size_t>(v);
    return true;
  };
  std::string truncated;  // the block the file ends inside, if any
  bool complete_structure = false;
  bool is_md = false;

  Structure s;
  // Ionic steps, counted where they end: an MD step at its EKIN_LAT line,
  // a relaxation step at its free-energy line. The label numbers them.
  size_t ionic_step = 0;
  bool have_energy = false;  // this step's first-principles free energy
  bool have_stress = false;  // this step's 'in kB' line
  // The step being read has started: its first SCF iteration line, or its
  // stress or TOTAL-FORCE block (the header's lattice comes before any
  // step's iterations, and each step's lattice after them).
  bool step_started = false;
  size_t stored = 0, without_stress = 0;

  // Steps of VASP's machine-learned force field (ML_MODE = train/run) are
  // not first-principles data. Their blocks are marked 'ML' ('ML FORCE on
  // cell', 'TOTAL-FORCE (eV/Angst) (ML)', 'free  energy ML TOTEN'); they are
  // never read, and the steps are skipped with one warning at the end.
  bool ml_in_step = false;
  bool ml_stress_block = false;  // the next 'in kB' line is an ML one
  std::vector<size_t> skipped_ml_steps;

  // Starts the next step from scratch; the label so far (the file name,
  // set from the header for the first structure) is kept.
  auto discard_step = [&]() {
    const std::string label = s.label;
    s = Structure();
    s.label = label;
    have_energy = false;
    have_stress = false;
    ml_in_step = false;
    step_started = false;
  };

  while (next_line()) {
    // '----- Iteration      2(   1)  -----': an SCF iteration of a
    // first-principles step.
    if (line.find("- Iteration ") != std::string::npos &&
        line.find('(') != std::string::npos)
      step_started = true;

    if (line.find("molecular dynamics") != std::string::npos) {
      is_md = true;
      s.label += "MD ";
    }
    else if (line.find("VRHFIN") != std::string::npos) {
      std::string type = line.substr(line.find("=") + 1);
      type = tadah::core::trim(type.substr(0, type.find(":")));
      try {
        elements.push_back(tadah::core::PeriodicTable::find_by_symbol(type));
      } catch (const std::exception &) {
        refuse("VRHFIN names '" + type + "', which is not an element symbol");
      }
      atom_types.push_back(type);
      s.label += filename_ + " ";
    }

    else if (line.find("NIONS") != std::string::npos) {
      // '... number of ions     NIONS =     54': the word after 'NIONS ='.
      std::istringstream iss(line.substr(line.find("NIONS")));
      std::string key, eq, value, rest;
      if (!(iss >> key >> eq >> value) || eq != "=" || (iss >> rest) ||
          !count_of(value, natoms) || natoms == 0)
        refuse("cannot read the number of ions NIONS from " + quoted());
      have_natoms = true;
    }

    else if (line.find("ions per type = ") != std::string::npos) {
      // 'ions per type =  <n_1> ... <n_K>', one count per VRHFIN element.
      if (atom_types.empty())
        refuse("'ions per type' comes before any VRHFIN line, which names "
               "the elements");
      std::istringstream iss(line.substr(line.find('=') + 1));
      std::vector<std::string> counts;
      for (std::string w; iss >> w;) counts.push_back(w);
      natom_types.clear();
      bool ok = counts.size() == atom_types.size() && !counts.empty();
      for (size_t i = 0; ok && i < counts.size(); ++i) {
        size_t n = 0;
        ok = count_of(counts[i], n);
        if (ok) natom_types.push_back(n);
      }
      if (!ok)
        refuse("'ions per type' must give one count for each of the " +
               std::to_string(atom_types.size()) + " VRHFIN elements, got " +
               quoted());
      const size_t total = std::accumulate(natom_types.begin(), natom_types.end(),
                                           static_cast<size_t>(0));
      if (!have_natoms || total != natoms)
        refuse("the ions per type add up to " + std::to_string(total) +
               (have_natoms ? ", but NIONS is " + std::to_string(natoms)
                            : ", but no NIONS line comes before"));
    }

    else if (line.find("FORCE on cell") != std::string::npos) {
      ml_stress_block = line.find("ML FORCE on cell") != std::string::npos;
      if (ml_stress_block) ml_in_step = true;
    }

    else if (line.find("in kB") != std::string::npos && ml_stress_block) {
      ml_stress_block = false;
    }

    else if (line.find("in kB") != std::string::npos) {
      step_started = true;
      std::vector<double> row(6); // xx yy zz xy yz zx
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5])) {
        // An unreadable tensor (e.g. VASP's '*****' overflow) is no stress.
        std::cerr << "Warning: VASP OUTCAR " << filename_
                  << ": the stress line '" << line
                  << "' cannot be read; the step is stored without stress"
                  << std::endl;
        s.stress.set_zero();
        have_stress = false;
      } else {
        have_stress = true;
        s.stress(0,0) = row[0], s.stress(1,1) = row[1], s.stress(2,2) = row[2];
        s.stress(0,1) = s.stress(1,0) = row[3]; // xy
        s.stress(1,2) = s.stress(2,1) = row[4]; // yz
        s.stress(0,2) = s.stress(2,0) = row[5]; // zx
        // VASP prints the pressure-signed tensor (positive under
        // compression); Tadah stores the Cauchy stress, positive under
        // tension: negate, and convert kB to eV/A^3.
        s.stress *= -s_conv;
      }
    }

    else if (line.find("direct lattice vectors") != std::string::npos) {
      // OUTCAR lists each lattice vector a_i on its own row (then the
      // reciprocal vector); store as row i of cell.
      for (int i = 0; i < 3; ++i) {
        if (!next_line()) {
          truncated = "lattice vectors";
          break;
        }
        std::istringstream iss(line);
        if (!(iss >> s.cell(i,0) >> s.cell(i,1) >> s.cell(i,2)))
          refuse("lattice vector " + std::to_string(i + 1) + " of ionic step " +
                 std::to_string(ionic_step + 1) + ", " + quoted() +
                 ", is not three numbers");
      }
      if (!truncated.empty()) break;
    }

    else if (line.find("TOTAL-FORCE") != std::string::npos &&
             line.find("ML") != std::string::npos) {
      ml_in_step = true;
    }

    else if (line.find("TOTAL-FORCE (eV/Angst)") != std::string::npos) {
      step_started = true;
      if (natom_types.empty())
        refuse("a TOTAL-FORCE block before the 'ions per type' line");
      // The dashed line, then one 'x y z fx fy fz' line per atom, in the
      // order of the VRHFIN elements.
      if (!next_line()) {
        truncated = "TOTAL-FORCE block";
        break;
      }
      size_t count = 0;
      for (size_t j = 0; j < natom_types.size() && truncated.empty(); ++j) {
        for (size_t i = 0; i < natom_types[j]; ++i) {
          if (!next_line()) {
            truncated = "TOTAL-FORCE block";
            break;
          }
          std::istringstream iss(line);
          double px,py,pz,fx,fy,fz;
          if (!(iss >> px >> py >> pz >> fx >> fy >> fz))
            refuse("the TOTAL-FORCE line of atom " + std::to_string(count + 1) +
                   " of ionic step " + std::to_string(ionic_step + 1) + ", " +
                   quoted() + ", is not 'x y z fx fy fz'");
          s.add_atom(Atom(elements[j], px, py, pz, fx, fy, fz));
          count++;
        }
      }
      if (!truncated.empty()) break;
    }

    // The free energy F (TOTEN) of the ionic step, the energy whose
    // gradient the forces and the stress are. The ionic-step line has two
    // spaces between the words ('free  energy   TOTEN'), the per-SCF lines
    // one ('free energy    TOTEN'). A relaxation's structure is complete
    // here; an MD step's at its EKIN_LAT line.
    else if (line.find("free  energy   TOTEN  =") != std::string::npos) {
      std::istringstream iss(line);
      std::string tmp;
      if (!(iss >> tmp >> tmp >> tmp >> tmp >> s.energy)) {
        refuse("cannot read the free energy of ionic step " +
               std::to_string(ionic_step + 1) + " from " + quoted());
      }
      else {
        have_energy = true;
        if (!is_md) {
          complete_structure = true;
          s.label += "Structure " + std::to_string(++ionic_step);
        }
      }
    }
    else if (line.find("free  energy ML TOTEN") != std::string::npos) {
      ml_in_step = true;
      if (!is_md) {  // a relaxation step of the force field ends here
        skipped_ml_steps.push_back(++ionic_step);
        discard_step();
      }
    }
    else if (line.find("EKIN_LAT=") != std::string::npos) {
      // '... EKIN_LAT=  0.000000  (temperature  300.00 K)': VASP's fixed
      // width joins the number to the word from 10000 K up
      // ('(temperature10147.82 K)').
      const std::size_t t = line.find("temperature");
      std::istringstream iss(t == std::string::npos
                                 ? std::string()
                                 : line.substr(t + std::string("temperature").size()));
      if (!(iss >> s.T)) {
        refuse("cannot read the temperature of ionic step " +
               std::to_string(ionic_step + 1) + " from " + quoted());
      }
      else if (have_energy) {
        complete_structure = true;
        s.label += "Structure " + std::to_string(++ionic_step);
      }
      else if (ml_in_step) {
        skipped_ml_steps.push_back(++ionic_step);
        discard_step();
      }
      else {
        throw std::runtime_error(
            "VASP OUTCAR " + filename_ + ": ionic step " +
            std::to_string(ionic_step + 1) +
            " has no 'free  energy   TOTEN' line before its EKIN_LAT line "
            "(an incomplete step, or one this reader does not recognise)");
      }
    }

    if (complete_structure) {
      // A step without a stress tensor: zero stress and stress priority 0,
      // so that a STRESS fit ignores it.
      if (!have_stress) {
        s.stress.set_zero();
        s.sweight = 0.0;
        ++without_stress;
      }
      stdb.add(s);
      ++stored;
      complete_structure = false;
      s = Structure();
      have_energy = false;
      have_stress = false;
      ml_in_step = false;
      step_started = false;
    }

  }

  // A run stopped while writing a block of a step: the step is skipped.
  if (!truncated.empty()) {
    std::cerr << "WARNING: VASP OUTCAR " << filename_
              << ": the file ends inside the " << truncated << " of ionic step "
              << ionic_step + 1 << " (line " << line_no
              << "), which was skipped; " << stored << " structure(s) read."
              << std::endl;
  }
  // An MD run cut off after a step's free energy, before its EKIN_LAT line.
  else if (is_md && have_energy) {
    std::cerr << "WARNING: VASP OUTCAR " << filename_ << ": ionic step "
              << ionic_step + 1
              << ", the last, has no EKIN_LAT line (the file ends inside it) "
                 "and was skipped; "
              << stored << " structure(s) read." << std::endl;
  }
  // A run cut off inside a step but outside its blocks: during its SCF
  // iterations, or between its blocks, before its free energy.
  else if (step_started) {
    std::cerr << "WARNING: VASP OUTCAR " << filename_
              << ": the file ends inside ionic step " << ionic_step + 1
              << " (line " << line_no
              << "), which was skipped (it has no 'free  energy   TOTEN' "
                 "line); "
              << stored << " structure(s) read." << std::endl;
  }
  log_missing_stress("VASP OUTCAR " + filename_, without_stress, stored);

  if (!skipped_ml_steps.empty()) {
    const size_t shown = std::min<size_t>(skipped_ml_steps.size(), 10);
    std::ostringstream steps;
    for (size_t i = 0; i < shown; ++i)
      steps << (i ? ", " : "") << skipped_ml_steps[i];
    if (shown < skipped_ml_steps.size()) steps << ", ...";
    const bool one = skipped_ml_steps.size() == 1;
    std::cerr << "WARNING: VASP OUTCAR " << filename_ << ": "
              << (one ? "ionic step " : "ionic steps ") << steps.str();
    if (!one) std::cerr << " (" << skipped_ml_steps.size() << " steps)";
    std::cerr << (one ? " was" : " were")
              << " computed by VASP's machine-learned force field (marked ML) "
                 "and skipped: only first-principles steps are stored."
              << std::endl;
  }
}

void VaspOutcarReader::print_summary() const {
  std::cout << get_summary();
}

std::string VaspOutcarReader::get_summary() const {
  return stdb.summary();
}
} // namespace mlip
} // namespace tadah
