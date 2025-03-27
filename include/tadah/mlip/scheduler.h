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
 * The Scheduler collects Task objects (via shared_ptr),
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
   * @param task A shared_ptr to the Task.
   */
  void addTask(const std::shared_ptr<Task> &task);

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
   * Execution is simulated by printing task details.
   */
  void execute();

  /**
   * @brief Sets a dependency rule (stub) for task ordering.
   *
   * @param rule A string representing the dependency rule.
   */
  void setDependencyRule(const std::string &rule);

  /**
   * @brief Serializes all tasks to a string.
   *
   * Used for debugging and MPI broadcast.
   *
   * @return A string containing the serialized tasks.
   */
  std::string serializeTasks() const;

private:
  std::vector<std::shared_ptr<Task>> tasks_;
  std::string dependencyRule_; ///< Stub for dependency ordering.
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_SCHEDULER_H

