#include <tadah/mlip/engines/analysis_engine.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

namespace tadah {
namespace mlip {
namespace engines {

int AnalysisEngine::descriptor(const std::vector<std::string>& datasets, const std::string &pot_file,
                               const std::string &outfile, const std::vector<std::string>& indices) {
  std::cout << "AnalysisEngine: Calculating structure descriptors (stub)." << std::endl;
  return EXIT_SUCCESS;
}

/**
 * @brief Evaluates and plots basis functions.
 *
 * Implements a dummy function: B(x) = sin(x) (or cos(x) if derivative is set),
 * multiplied by 0.5 if scale is enabled.
 * Writes output to the specified file.
 */
int AnalysisEngine::bfunc(const std::string &context_file, const std::string &outfile,
  const std::vector<double>& range, const std::vector<std::string>& types,
  bool derivative, bool scale, int numeric_precision) {
  std::ofstream ofs(outfile);
  if (!ofs.is_open()) {
    std::cerr << "AnalysisEngine::bfunc: Cannot open outfile." << std::endl;
    return EXIT_FAILURE;
  }
  double start = (range.size() >= 1) ? range[0] : 0.0;
  double stop  = (range.size() >= 2) ? range[1] : 5.0;
  int npoints  = (range.size() >= 3) ? static_cast<int>(range[2]) : 100;
  double dx = (stop - start) / (npoints - 1);
  for (int i = 0; i < npoints; i++) {
    double x = start + i * dx;
    double value = derivative ? std::cos(x) : std::sin(x);
    if (scale)
      value *= 0.5;
    ofs << x << " " << value << "\n";
  }
  ofs.close();
  std::cout << "AnalysisEngine::bfunc: Basis functions written to " << outfile << std::endl;
  return EXIT_SUCCESS;
}

/**
 * @brief Evaluates and plots cutoff functions.
 *
 * Implements a dummy cutoff function:
 * C(x) = 0.5 * (1 + cos(x)) for x in [0, pi], 0 otherwise.
 * If derivative is set, outputs dC/dx = -0.5 * sin(x) in [0,pi].
 * Results are written to outfile.
 */
int AnalysisEngine::cutoff(const std::string &outfile, const std::vector<double>& range,
  const std::vector<std::string>& types, bool derivative, int numeric_precision) {
  std::ofstream ofs(outfile);
  if (!ofs.is_open()) {
    std::cerr << "AnalysisEngine::cutoff: Cannot open outfile." << std::endl;
    return EXIT_FAILURE;
  }
  double start = (range.size() >= 1) ? range[0] : 0.0;
  double stop  = (range.size() >= 2) ? range[1] : 3.14159;
  int npoints  = (range.size() >= 3) ? static_cast<int>(range[2]) : 100;
  double dx = (stop - start) / (npoints - 1);
  for (int i = 0; i < npoints; i++) {
    double x = start + i * dx;
    double value = 0.0;
    if (x >= 0.0 && x <= 3.14159) {
      value = derivative ? -0.5 * std::sin(x) : 0.5 * (1 + std::cos(x));
    }
    ofs << x << " " << value << "\n";
  }
  ofs.close();
  std::cout << "AnalysisEngine::cutoff: Cutoff functions written to " << outfile << std::endl;
  return EXIT_SUCCESS;
}

} // namespace engines
} // namespace mlip
} // namespace tadah

