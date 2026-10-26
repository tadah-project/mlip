#include <tadah/mlip/task.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <tadah/core/logger.h>

namespace tadah {
namespace mlip {

//-------------------------------------------------------------------------
// Constructor
//-------------------------------------------------------------------------
Task::Task(const std::string &command, const tadah::core::Context &context)
  : command_(command),
    priority_(0),
    taskID_(generateTaskID()),
    timestamp_(generateTimestamp()),
    context_(context) // copy the passed context
{
}

//-------------------------------------------------------------------------
// Public getters
//-------------------------------------------------------------------------
const tadah::core::Context &Task::getContext() const {
  return context_;
}
tadah::core::Context &Task::getContext() {
  return context_;
}

std::string Task::getID() const {
  return taskID_;
}

std::string Task::getTimestamp() const {
  return timestamp_;
}

int Task::getPriority() const {
  return priority_;
}

void Task::setKeyProvenance(KeyProvenance provenance) {
  provenance_ = std::move(provenance);
}

const KeyProvenance *Task::getKeyProvenance() const {
  return provenance_ ? &*provenance_ : nullptr;
}

//-------------------------------------------------------------------------
// Public setters
//-------------------------------------------------------------------------
void Task::setPriority(int p) {
  priority_ = p;
}

// void Task::setDependency(const std::string &dependencyRule) {
//   dependencyRule_ = dependencyRule;
// }

//-------------------------------------------------------------------------
// generateTaskID(): simplistic ID generator
//-------------------------------------------------------------------------
std::string Task::generateTaskID() const {
  static int counter = 0;
  ++counter;
  std::ostringstream oss;
  oss << "TASK-" << counter << "-"
      << std::chrono::system_clock::now().time_since_epoch().count();
  return oss.str();
}

//-------------------------------------------------------------------------
// generateTimestamp(): near-ISO8601
//-------------------------------------------------------------------------
std::string Task::generateTimestamp() const {
  auto now     = std::chrono::system_clock::now();
  std::time_t t= std::chrono::system_clock::to_time_t(now);

  // Thread-safe localtime
  std::tm tmBuf;
#ifdef _WIN32
  localtime_s(&tmBuf, &t);
#else
  localtime_r(&t, &tmBuf);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tmBuf, "%Y-%m-%dT%H:%M:%S");
  return oss.str();
}

//-------------------------------------------------------------------------
// print(): for quick debug
//-------------------------------------------------------------------------
void Task::print() const {
  std::cout << "Task ID: " << taskID_ << "\n"
            << "Command: " << command_ << "\n"
            << "Timestamp: " << timestamp_ << "\n"
            << "Priority: " << priority_ << "\n"
            << "Context:\n";

  // Print the context
  std::cout << context_ << std::endl;
}

//-------------------------------------------------------------------------
// operator<<()
//-------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const Task &t) {
  os << "[Task: ID=" << t.taskID_
     << ", Command=" << t.command_
     << ", Timestamp=" << t.timestamp_
     << ", Priority=" << t.priority_ 
     << "]";
  return os;
}

//-------------------------------------------------------------------------
// getTaskName()
//-------------------------------------------------------------------------
std::string Task::getTaskName() const {
  return command_;
}

} // namespace mlip
} // namespace tadah
