#ifndef TADAH_MLIP_SCHEDULER_H
#define TADAH_MLIP_SCHEDULER_H

#include <string>
#include <vector>
#include <memory>
#include <tadah/mlip/task.h>

namespace tadah {
namespace mlip {

/**
 * @brief Manages and executes tasks.
 *
 * The Scheduler collects Task objects,
 * orders them based on original order, global priority, and task type groupings.
 * A stub for dependency rules is provided.
 */
class Scheduler {
public:
  /**
   * @brief Constructs a Scheduler.
   */
  Scheduler();

  /**
   * @brief Adds a Task to the scheduler.
   *
   * @param task the Task.
   */
  void addTask(const Task &task);
  void addTasks(std::vector<Task> tasks);

  /**
   * @brief Schedules tasks.
   *
   * The default ordering is the order of appearance in the user_config.
   * Secondary sorting is done by global priority and grouping by task type.
   */
  void schedule();

  /**
   * @brief Executes scheduled tasks sequentially.
   *
   * Before it runs any task it checks that each has an engine
   * (engines::EngineFactory::require), so a task without one, such as a
   * task the program was built without, is refused before an earlier task
   * runs.
   *
   * @return 0 when every task's engine returned 0; otherwise the status
   *         returned by the first failing engine, after which the
   *         remaining tasks are not run.
   * @throws std::runtime_error what EngineFactory::require throws for the
   *         first task without an engine.
   */
  int execute();

  /**
   * @brief Sets a dependency rule (stub) for task ordering.
   *
   * @param rule A string representing the dependency rule.
   */
  // void setDependencyRule(const std::string &rule);

private:
  std::vector<Task> tasks_;
  // std::string dependencyRule_; ///< Stub for dependency ordering.
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_SCHEDULER_H

