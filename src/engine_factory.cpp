#include <tadah/mlip/engines/engine_factory.h>
#include <stdexcept>

namespace tadah {
namespace mlip {
namespace engines {

static std::unordered_map<std::string, std::function<std::unique_ptr<IEngine>()>>&
theRegistry() {
  static std::unordered_map<std::string, std::function<std::unique_ptr<IEngine>()>> reg;
  return reg;
}

std::unique_ptr<IEngine> EngineFactory::create(const std::string &taskName) {
  auto &reg = theRegistry();
  auto it = reg.find(taskName);
  if (it == reg.end()) {
    throw std::runtime_error("EngineFactory::create: No engine found for " + taskName);
  }
  return it->second();
}

void EngineFactory::registerEngine(const std::string &taskName,
  std::function<std::unique_ptr<IEngine>()> builder) {
  theRegistry()[taskName] = builder;
}

std::unordered_map<std::string, std::function<std::unique_ptr<IEngine>()>>&
EngineFactory::registry() {
  return theRegistry();
}

} // namespace engines
} // namespace mlip
} // namespace tadah
