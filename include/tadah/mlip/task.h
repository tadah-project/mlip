#ifndef TADAH_MLIP_TASK_H
#define TADAH_MLIP_TASK_H

#include <string>
#include <memory>
#include <iostream>
#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/**
 * @brief Represents a task created from a user configuration.
 *
 * A Task stores a command, unique ID, timestamp, priority, and an effective Context
 * that merges global settings with task-specific overrides. Serialization is provided
 * in both plain-text and binary formats.
 */
class Task {
public:
  /**
   * @brief Constructs a Task with a command and global Context.
   * @param command The task command (e.g., "data convert").
   * @param globalContext The global configuration context.
   */
  Task(const std::string &command, const tadah::core::Context &globalContext);

  /**
   * @brief Returns the effective Context (merged global and task-specific settings).
   * @return A constant reference to the Context.
   */
  const tadah::core::Context& effectiveContext() const;

  /**
   * @brief Sets the task priority.
   * @param p The priority (higher means higher priority).
   */
  void setPriority(int p);

  /**
   * @brief Retrieves the task priority.
   * @return The task priority.
   */
  int getPriority() const;

  /**
   * @brief Retrieves the unique task ID.
   * @return The task ID as a string.
   */
  std::string getID() const;

  /**
   * @brief Retrieves the timestamp of task creation.
   * @return The timestamp as a string.
   */
  std::string getTimestamp() const;

  /**
   * @brief Serializes the task to a plain-text string.
   * @return Serialized task data.
   */
  std::string serialize() const;

  /**
   * @brief Serializes selected fields to a binary format.
   *
   * The binary format includes fields such as ID, task type, and timestamp.
   * @return A vector of bytes representing the serialized data.
   */
  std::vector<char> serializeBinary() const;

  /**
   * @brief Deserializes a Task from a plain-text string.
   * @param data Serialized task data.
   * @param globalContext Global context to merge with.
   * @return The deserialized Task.
   * @throws std::runtime_error on format errors.
   */
  static Task deserialize(const std::string &data, const tadah::core::Context &globalContext);

  /**
   * @brief Prints the task details to standard output.
   */
  void print() const;

  /**
   * @brief Overloads operator<< to output task information.
   * @param os Output stream.
   * @param t The Task object.
   * @return The output stream.
   */
  friend std::ostream& operator<<(std::ostream &os, const Task &t);

  /**
   * @brief Sets a dependency rule (stub). 
   *
   * Dependency rules specify order relationships (e.g., "A>B").
   * Currently a stub for future implementation.
   *
   * @param dependencyRule A string representing the dependency.
   */
  void setDependency(const std::string &dependencyRule);

  /**
   * @brief Retrieves the task type.
   *
   * The task type is extracted from the command (typically the first word).
   * @return A string representing the task type.
   */
  std::string getTaskType() const;

private:
  std::string command_;
  int priority_;
  std::string taskID_;
  std::string timestamp_;
  tadah::core::Context context_; ///< Merged context
  std::string dependencyRule_;    ///< Current dependency rule (stub)

  /**
   * @brief Generates a unique task ID.
   * @return A string representing a unique task ID.
   */
  std::string generateTaskID() const;

  /**
   * @brief Generates the current timestamp.
   * @return A string representing the timestamp.
   */
  std::string generateTimestamp() const;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_TASK_H

