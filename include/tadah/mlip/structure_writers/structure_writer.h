#ifndef STRUCTURE_WRITER_H
#define STRUCTURE_WRITER_H

#include <tadah/mlip/structure_db.h>
#include <tadah/core/number_format.h>

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>
namespace tadah {
namespace mlip {

class StructureWriter {
  public:
    virtual ~StructureWriter() = default;

    virtual void write_data(const std::string& filename, const size_t i) = 0;

    StructureWriter(StructureDB& db) : stdb(db) {};

    virtual void set_precision(const size_t _p) { p = _p; w = p+6; };

  protected:
    StructureDB& stdb;
    double p = 18;  // output precision (NUMERIC; default 18)
    double w = p+6;  // column width

    /** Writes @p v with p SIGNIFICANT digits (%.{N}g,
     *  core::format_sig_chars; p = 18 is exact), right-aligned in a column
     *  one wider than the widest rendering, so consecutive columns are
     *  always separated by a space (@p pad false: no padding). For formats
     *  whose readers accept E-notation reals (LAMMPS data files, VASP
     *  POSCAR). */
    void put_sig(std::ostream &os, double v, bool pad = true) const {
      const int digits =
          std::clamp(static_cast<int>(p), 1, tadah::core::kMaxSigDigits);
      char buf[tadah::core::kReformatBufferSize];
      const std::size_t n =
          tadah::core::format_sig_chars(v, digits, buf, sizeof buf);
      if (pad) os << std::right << std::setw(digits + 8);
      os << std::string_view(buf, n);
    }
};
} // namespace mlip
} // namespace tadah
#endif // STRUCTURE_WRITER_H
