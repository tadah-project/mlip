#ifndef TADAH_MLIP_ENGINES_REFIT_LOG_H
#define TADAH_MLIP_ENGINES_REFIT_LOG_H

/** \file refit_log.h
 *  \brief The `tadah refit` optimisation log (REFIT_LOG, default
 *  refit_opt.log): a per-run, per-line-flushed progress file the user can
 *  inspect DURING the fit — the refit counterpart of the HPO logger,
 *  adapted to this optimisation task. The compact engine writes the header,
 *  one line per ladder event (seed / rung / coalescence merge / final
 *  optimisation), the joint-refinement summary and the crystal self-test;
 *  the dense path writes its escalation loop and goodness-of-fit.
 */

#include <ctime>
#include <fstream>
#include <string>

namespace tadah {
namespace mlip {
namespace engines {

class RefitLog {
public:
  void open(const std::string &path, bool append) {
    os_.open(path, append ? std::ios::app : std::ios::trunc);
  }
  bool ok() const { return os_.is_open() && os_.good(); }
  /** One timestamped line, flushed immediately (live inspection). */
  void line(const std::string &msg) {
    if (!ok()) return;
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&t));
    os_ << "[" << buf << "] " << msg << "\n";
    os_.flush();
  }
  /** Raw (untimestamped) line — for tables and headers. */
  void raw(const std::string &msg) {
    if (!ok()) return;
    os_ << msg << "\n";
    os_.flush();
  }

private:
  std::ofstream os_;
};

}  // namespace engines
}  // namespace mlip
}  // namespace tadah
#endif  // TADAH_MLIP_ENGINES_REFIT_LOG_H
