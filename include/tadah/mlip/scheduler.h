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
 * Stubs for dependency rules are provided along with serialization for MPI broadcast.
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
   * @return 0 when every task's engine returned 0; otherwise the status
   *         returned by the first failing engine, after which the
   *         remaining tasks are not run.
   */
  int execute();

  /**
   * @brief Sets a dependency rule (stub) for task ordering.
   *
   * @param rule A string representing the dependency rule.
   */
  // void setDependencyRule(const std::string &rule);

  /**
   * @brief Broadcast tasks_ from rank 0 to all other ranks,
   *        ensuring all ranks see the same tasks in the same order.
   */
  void distributeTasks();

private:
  std::vector<Task> tasks_;
  // std::string dependencyRule_; ///< Stub for dependency ordering.
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_SCHEDULER_H

