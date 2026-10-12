#ifndef TADAH_MLIP_ENGINES_ENGINE_FACTORY_H
#define TADAH_MLIP_ENGINES_ENGINE_FACTORY_H

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>

#include <tadah/mlip/engines/iengine.h>

namespace tadah {
namespace mlip {
namespace engines {

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
   * @throws std::runtime_error with the message registerNotBuilt() gave
   *         when the program was built without the task's engine;
   *         "EngineFactory::create: No engine found for <taskName>" when
   *         nothing is registered for it.
   */
  static std::unique_ptr<IEngine> create(const std::string &taskName);

  /**
   * @brief Throws what create() would throw for @p taskName, without
   *        building an engine; returns when an engine is registered.
   *
   * Scheduler::execute() calls it for every task before it runs any.
   */
  static void require(const std::string &taskName);

  /**
   * @brief Registers an engine builder for a given task name.
   * @param taskName The dot-separated name (e.g. "data.merge").
   * @param builder A callable returning a new IEngine instance.
   *
   * It replaces a builder, or a registerNotBuilt() entry, of that name.
   */
  static void registerEngine(const std::string &taskName,
    std::function<std::unique_ptr<IEngine>()> builder);

  /**
   * @brief Records that this program was built without the engine of
   *        @p taskName, and why.
   *
   * create() and require() then throw std::runtime_error(@p message), and
   * `tadah -h`, `tadah <command> -h` and `tadah explain <command>` show it
   * (Tadah.MLIP registers HPO so in a tadah built without
   * TADAH_ENABLE_NFP). The message is the whole refusal, without a full
   * stop: "tadah hpo needs ..., which this tadah was built without;
   * reconfigure with ...". It replaces a builder of that name.
   */
  static void registerNotBuilt(const std::string &taskName,
                               const std::string &message);

  /**
   * @brief The message registerNotBuilt() recorded for @p taskName; empty
   *        when an engine is registered for it, or nothing is.
   */
  static std::string notBuiltMessage(const std::string &taskName);

private:
  static std::unordered_map<std::string, std::function<std::unique_ptr<IEngine>()>>& registry();
};

} // namespace engines
} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_ENGINES_ENGINE_FACTORY_H
