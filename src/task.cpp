#include <tadah/mlip/task.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <stdexcept>
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

std::string Task::getID() const {
  return taskID_;
}

std::string Task::getTimestamp() const {
  return timestamp_;
}

int Task::getPriority() const {
  return priority_;
}

//-------------------------------------------------------------------------
// Public setters
//-------------------------------------------------------------------------
void Task::setPriority(int p) {
  priority_ = p;
}

void Task::setDependency(const std::string &dependencyRule) {
  dependencyRule_ = dependencyRule;
}

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
// serialize(): text-based, uses context_.serialize()
//-------------------------------------------------------------------------
std::string Task::serialize() const {
  // We'll store metadata as lines, then the Context lines below.
  // Format:
  // TASKID <id>
  // COMMAND <cmd>
  // TIMESTAMP <ts>
  // PRIORITY <n>
  // CONTEXT
  //   <lines from context serialization>
  //
  // Note: The context lines themselves don't have a prefix. We'll just store them
  // after a line that says "CONTEXT". The user can parse until next blank line or EOF.

  std::ostringstream oss;
  oss << "TASKID " << taskID_ << "\n"
      << "COMMAND " << command_ << "\n"
      << "TIMESTAMP " << timestamp_ << "\n"
      << "PRIORITY " << priority_ << "\n"
      << "CONTEXT\n";

  // The context is turned into lines of "KEY val1 val2 ..."
  std::vector<char> buffer = context_.serialize();
  // Convert those chars back into string lines
  oss.write(buffer.data(), buffer.size());

  // (Optional) A trailing blank line
  oss << "\n";
  return oss.str();
}

//-------------------------------------------------------------------------
// deserialize(): parses lines, then uses context_.deserialize()
//-------------------------------------------------------------------------
Task Task::deserialize(const std::string &data,
                       const tadah::core::Context &baseContext) {
  // We'll create a new Context from baseContext. Then we overwrite it from
  // incoming lines. The lines after "CONTEXT" are fed to the context's deserialize().

  std::istringstream iss(data);
  std::string line;

  std::string id;
  std::string cmd;
  std::string ts;
  int pri = 0;

  tadah::core::Context ctx(baseContext); // copy baseline

  bool readingContext = false;
  std::ostringstream ctxStream;

  while (std::getline(iss, line)) {
    // Trim
    if (line.size() == 0) {
      continue; // ignore blank line
    }
    // If we see "CONTEXT", subsequent lines belong to the context
    if (!readingContext) {
      // parse "KEY value"
      std::istringstream parts(line);
      std::string key;
      parts >> key; 
      if (key == "TASKID") {
        parts >> id;
      } else if (key == "COMMAND") {
        parts >> cmd;
      } else if (key == "TIMESTAMP") {
        parts >> ts;
      } else if (key == "PRIORITY") {
        parts >> pri;
      } else if (key == "CONTEXT") {
        // Now we start capturing context lines
        readingContext = true;
      }
      // otherwise ignore unknown lines
    } else {
      // We are reading context lines
      ctxStream << line << "\n";
    }
  }

  if (id.empty()) {
    throw std::runtime_error("Task::deserialize - missing TASKID");
  }
  if (cmd.empty()) {
    throw std::runtime_error("Task::deserialize - missing COMMAND");
  }

  // Now that we have the context lines, let's feed them to ctx
  // We do so by building a vector<char>
  const std::string ctxStr = ctxStream.str();
  std::vector<char> contextData(ctxStr.begin(), ctxStr.end());
  ctx.deserialize(contextData);

  // Construct final Task. We'll override its ID/timestamp/priority from file
  Task t(cmd, ctx);
  t.taskID_    = id;
  t.timestamp_ = ts;
  t.priority_  = pri;
  return t;
}

//-------------------------------------------------------------------------
// serializeBinary(): minimal example
//-------------------------------------------------------------------------
std::vector<char> Task::serializeBinary() const {
  // Example: store ID, command, timestamp, priority. 
  // The context could also be appended, e.g. as text or more advanced structure.

  std::vector<char> bytes;

  auto writeString = [&](const std::string &s) {
    bytes.insert(bytes.end(), s.begin(), s.end());
    bytes.push_back('\0'); // null-terminator
  };

  writeString(taskID_);
  writeString(command_);
  writeString(timestamp_);

  // Priority as 4 bytes
  int localP = priority_;
  char* pBytes = reinterpret_cast<char*>(&localP);
  for (size_t i = 0; i < sizeof(localP); i++) {
    bytes.push_back(pBytes[i]);
  }

  // (Optional) we could then store context_.serialize() as well.

  return bytes;
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
