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

// The engines this program was built without, with the message that says
// why (registerNotBuilt); a name is in at most one of the two registries.
static std::unordered_map<std::string, std::string>& theNotBuilt() {
  static std::unordered_map<std::string, std::string> reg;
  return reg;
}

void EngineFactory::require(const std::string &taskName) {
  if (theRegistry().count(taskName)) return;
  auto &nb = theNotBuilt();
  const auto it = nb.find(taskName);
  if (it != nb.end()) throw std::runtime_error(it->second);
  // TaskFactory refuses a task the blueprint does not know before.
  throw std::runtime_error("internal error: no engine runs the task " +
                           taskName);
}

std::unique_ptr<IEngine> EngineFactory::create(const std::string &taskName) {
  require(taskName);
  return theRegistry().at(taskName)();
}

void EngineFactory::registerEngine(const std::string &taskName,
  std::function<std::unique_ptr<IEngine>()> builder) {
  theNotBuilt().erase(taskName);
  theRegistry()[taskName] = builder;
}

void EngineFactory::registerNotBuilt(const std::string &taskName,
                                     const std::string &message) {
  theRegistry().erase(taskName);
  theNotBuilt()[taskName] = message;
}

std::string EngineFactory::notBuiltMessage(const std::string &taskName) {
  const auto &nb = theNotBuilt();
  const auto it = nb.find(taskName);
  return it == nb.end() ? std::string() : it->second;
}

std::unordered_map<std::string, std::function<std::unique_ptr<IEngine>()>>&
EngineFactory::registry() {
  return theRegistry();
}

} // namespace engines
} // namespace mlip
} // namespace tadah
