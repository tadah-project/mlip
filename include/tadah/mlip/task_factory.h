/**
 * @file task_factory.h
 * @brief Declares TaskFactory, responsible for converting ITaskInput into Task objects.
 *
 * TaskFactory uses MLIPUserConfigParser to handle config files specified
 * in the incoming task input. It merges user config data with the base
 * context to build fully populated Tasks.
 */

#ifndef MLIP_TASK_FACTORY_H
#define MLIP_TASK_FACTORY_H

#include <tadah/mlip/task.h>
#include <tadah/mlip/mlip_user_config_parser.h>
#include <tadah/mlip/itask_input.h>
#include <vector>
#include <memory>

namespace tadah {
namespace mlip {

/**
 * @class TaskFactory
 * @brief Creates one or more Task objects from ITaskInput sources.
 *
 * By depending on ITaskInput, TaskFactory remains unaware of any
 * front-end specifics. The optional MLIPUserConfigParser merges
 * config files (if specified) with the base key-value data from the input.
 */
class TaskFactory {
public:
  /**
   * @brief Constructs a TaskFactory.
   * @param parser Shared pointer to an MLIPUserConfigParser instance.
   *               Defaults to a fresh parser if none provided.
   */
  explicit TaskFactory(std::shared_ptr<MLIPUserConfigParser> parser
                       = std::make_shared<MLIPUserConfigParser>());
  /**
   * @brief Default destructor.
   */
  ~TaskFactory() = default;

  /**
   * @brief Creates tasks from an ITaskInput.
   *
   * Steps:
   *  0. Validate the key-value data against the blueprint schema.
   *  1. Build a base Context from the input's key-value data.
   *  2. Check if a `<taskName>.config` key exists; if so, parse and merge
   *     each config section with the base context.
   *  3. Construct one or more Task objects, returning them in a vector.
   *
   * @param taskInput The source of key-value data and the associated task name.
   * @return A vector of one or more Task objects.
   */
  std::vector<Task> createTasksFromInput(const ITaskInput &taskInput);

private:
  std::shared_ptr<MLIPUserConfigParser> parser_; ///< Shared parser for config files.
};

} // namespace mlip
} // namespace tadah

#endif // MLIP_TASK_FACTORY_H
