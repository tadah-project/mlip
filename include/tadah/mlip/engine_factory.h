#ifndef TADAH_MLIP_ENGINE_FACTORY_H
#define TADAH_MLIP_ENGINE_FACTORY_H

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include <tadah/mlip/iengine.h>

namespace tadah {
namespace mlip {

/**
 * @brief Simple dynamic factory that maps a dot-separated task name to a builder function.
 *
 * The builder function returns a std::unique_ptr<IEngine>. No Context is captured here.
 */
class EngineFactory {
public:
  /**
   * @brief Creates an engine for a given dot-separated task name.
   * @param taskName The lowercase or dot-notation name (e.g. "data.merge").
   * @return A unique_ptr to the newly created IEngine instance.
   * @throws std::runtime_error If no matching builder is found.
   */
  static std::unique_ptr<IEngine> create(const std::string &taskName);

  /**
   * @brief Registers an engine builder for a given task name.
   * @param taskName The dot-separated name (e.g. "data.merge").
   * @param builder A callable returning a new IEngine instance.
   */
  static void registerEngine(const std::string &taskName,
    std::function<std::unique_ptr<IEngine>()> builder);

private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<IEngine>()>>& registry();
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINE_FACTORY_H
