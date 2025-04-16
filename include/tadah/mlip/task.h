/**
 * @file task.h
 * @brief Declares the Task class which represents an individual task in Tadah!.
 *
 * A Task encapsulates:
 *  - A command (e.g. "data.split")
 *  - A unique task ID
 *  - A timestamp (creation time)
 *  - A priority
 *  - A merged (effective) Context object
 */

#ifndef TADAH_MLIP_TASK_H
#define TADAH_MLIP_TASK_H

#include <string>
#include <vector>
#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/**
 * @class Task
 * @brief Represents a user-defined operation in Tadah! with its own context.
 *
 * A Task is created after merging user configs and CLI data into a single Context.
 * It stores additional metadata like a unique identifier, timestamp, and priority.
 * This metadata aids in scheduling, logging, and debugging.
 */
class Task {
public:
  /**
   * @brief Constructs a Task with a command (e.g., "data.split") and a final merged Context.
   *
   * @param command The name or type of the task, for example "data.split".
   * @param context The fully merged configuration context for this task.
   */
  Task(const std::string &command, const tadah::core::Context &context);

  /**
   * @brief Default constructor.
   *
   * This constructor initializes an empty Task object.
   */
  Task() = default;

  /**
   * @brief Default destructor.
   */
  ~Task() = default;

  /**
   * @brief Retrieves the Context for this task.
   *
   * @return A constant reference to the Context representing the task's settings.
   */
  const tadah::core::Context &getContext() const;
  tadah::core::Context &getContext();

  /**
   * @brief Sets the integer priority for this task, where a higher number denotes higher priority.
   *
   * @param p The new priority value.
   */
  void setPriority(int p);

  /**
   * @brief Retrieves the current integer priority of this task.
   *
   * @return The task's priority.
   */
  int getPriority() const;

  /**
   * @brief Retrieves the unique ID generated for this task.
   *
   * @return A string containing the unique task ID.
   */
  std::string getID() const;

  /**
   * @brief Retrieves the timestamp at which this task was created.
   *
   * @return A string of the timestamp, e.g. "2023-10-05T14:23:59Z".
   */
  std::string getTimestamp() const;

  /**
   * @brief Serializes the task data to a plain-text string.
   *
   * Useful for logging or storing tasks externally.
   * @return A string containing task ID, command, timestamp, priority, and context data.
   */
  std::string serialize() const;

  /**
   * @brief Deserializes a Task from a plain-text string.
   *
   * @param data A string containing the serialized task data (e.g. from serialize()).
   * @return A newly constructed Task.
   * @throws std::runtime_error if the format is invalid.
   */
  void deserialize(const std::string &data);

  /**
   * @brief Serializes selected fields to a binary representation (e.g. for HPC distribution).
   *
   * @return A vector of bytes representing ID, command, timestamp, etc.
   */
  std::vector<char> serializeBinary() const;

  /**
   * @brief Deserializes a Task from a binary representation.
   *
   * @param data A string containing the serialized binary data.
   * @throws std::runtime_error if the format is invalid.
   */
  void deserializeBinary(const std::vector<char> &buffer);

  /**
   * @brief Prints debug information about the task to standard output.
   */
  void print() const;

  /**
   * @brief Stream insertion operator, prints summary info.
   * @param os Output stream
   * @param t The Task object
   * @return The output stream for chaining
   */
  friend std::ostream& operator<<(std::ostream &os, const Task &t);

  /**
   * @brief Retrieves the "task type," e.g. the first word or a substring of the command.
   *
   * @return A string representing the type, e.g. "data.split".
   */
  std::string getTaskName() const;

  /**
   * @brief Sets a dependency rule string for the task (e.g., "taskA > taskB").
   *
   * This is a placeholder for future dependency logic.
   * @param dependencyRule A string specifying the dependency requirement.
   */
  // void setDependency(const std::string &dependencyRule);

private:
  /**
   * @brief Generates a unique task ID (e.g., a UUID or random string).
   *
   * @return A string representing a generated unique identifier.
   */
  std::string generateTaskID() const;

  /**
   * @brief Generates a timestamp (e.g. ISO8601) to mark when the task was created.
   *
   * @return A string representing the task creation timestamp.
   */
  std::string generateTimestamp() const;

private:
  std::string command_;         ///< The full command string (e.g. "data.split").
  int priority_;                ///< The integer priority (default 0).
  std::string taskID_;          ///< The unique ID for this task.
  std::string timestamp_;       ///< The creation timestamp (ISO8601 or similar).
  tadah::core::Context context_;///< The merged context data for this task.
  // std::string dependencyRule_;  ///< Future expansion for visualizing dependencies.
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_TASK_H
