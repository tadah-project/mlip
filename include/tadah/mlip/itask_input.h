/**
 * @file itask_input.h
 * @brief Defines the ITaskInput interface for creating Tasks from arbitrary key-value data.
 *
 * The ITaskInput interface abstracts the notion of "task input," which
 * may come from CLI, config files, or any other source. By implementing
 * this interface, objects can hand off the necessary data to TaskFactory
 * without TaskFactory needing to know their concrete type.
 */

#ifndef MLIP_ITASK_INPUT_H
#define MLIP_ITASK_INPUT_H

#include <unordered_map>
#include <string>
#include <vector>

namespace tadah {
namespace mlip {

/**
 * @class ITaskInput
 * @brief Pure virtual interface that provides key-value input data.
 *
 * Implementations of ITaskInput must provide a task name and a set of
 * key-value pairs used to build a Context for the task.
 */
class ITaskInput {
public:
  /// Virtual destructor to allow polymorphic deletion.
  virtual ~ITaskInput() = default;

  /**
   * @brief Retrieves the canonical task name.
   *
   * For instance, "data.split" or "training.fit" might be returned here.
   *
   * @return A string representing the task name.
   */
  virtual std::string getTaskName() const = 0;

  /**
   * @brief Retrieves the key-value data needed by TaskFactory.
   *
   * The keys and values are later merged into a tadah::core::Context.
   *
   * @return A map of string keys to a vector of string values.
   */
  virtual std::unordered_map<std::string, std::vector<std::string>> getKeyValueData() const = 0;
};

} // namespace mlip
} // namespace tadah

#endif // MLIP_ITASK_INPUT_H
