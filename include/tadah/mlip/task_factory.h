/**
 * @file task_factory.h
 * @brief Declares TaskFactory, responsible for converting ITaskInput into Task objects.
 *
 * TaskFactory reads the configuration file an input gives (CONFIG, the -c
 * file of the command line) with CORE's configuration parser and merges
 * the input's own key-value data over it to build the Task of one command.
 */

#ifndef MLIP_TASK_FACTORY_H
#define MLIP_TASK_FACTORY_H

#include <tadah/mlip/task.h>
#include <tadah/mlip/itask_input.h>
#include <vector>

namespace tadah {
namespace mlip {

/**
 * @class TaskFactory
 * @brief Creates the Task of one command from an ITaskInput.
 *
 * By depending on ITaskInput, TaskFactory remains unaware of any
 * front-end specifics. A configuration file the input names (CONFIG) is
 * read with core::CoreUserConfigParser and its keys merged under the
 * input's own key-value data.
 */
class TaskFactory {
public:
  TaskFactory() = default;
  ~TaskFactory() = default;

  /**
   * @brief Creates the task of an ITaskInput.
   *
   * Steps:
   *  0. Validate the key-value data against the blueprint: a key that is
   *     no configuration key ("task" among them) is refused.
   *  1. Build a base Context from the input's key-value data.
   *  2. If the input gives CONFIG, parse that file (and the files it
   *     includes) with core::CoreUserConfigParser -- every line a key line,
   *     an unknown key an error at its line and file -- and merge the base
   *     context over it.
   *  3. Add the blueprint defaults of the command's options.
   *
   * @param taskInput The source of key-value data and the associated task name.
   * @return A vector of the one Task of the command.
   * @throws std::invalid_argument when the task name of @p taskInput is not
   *         a blueprint task, before any configuration file is read: the
   *         message names it, the task it matches but for the case of its
   *         letters (if one) and the task names.
   * @throws std::runtime_error for an invalid key or value, the name of a
   *         group of tasks (such as `DATA`), and a configuration file that
   *         cannot be read or holds a line that is not a key line.
   */
  std::vector<Task> createTasksFromInput(const ITaskInput &taskInput);
};

} // namespace mlip
} // namespace tadah

#endif // MLIP_TASK_FACTORY_H
