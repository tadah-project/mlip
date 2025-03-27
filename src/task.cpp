/*#include <tadah/mlip/task.h>*/
/*#include <tadah/core/logger.h>*/
/*#include <sstream>*/
/*#include <chrono>*/
/*#include <ctime>*/
/*#include <iomanip>*/
/*#include <stdexcept>*/
/*#include <vector>*/
/**/
/*namespace tadah {*/
/*namespace mlip {*/
/**/
/*using tadah::core::Context;*/
/**/
/*Task::Task(const std::string &command, const Context &globalContext)*/
/*  : command_(command), priority_(0), context_(globalContext)*/
/*{*/
/*  taskID_ = generateTaskID();*/
/*  timestamp_ = generateTimestamp();*/
/*}*/
/**/
/*void Task::setOption(const std::string &key, const std::string &value) {*/
/*  // If key already exists in the task-specific part, log a warning if scalar type substitution occurs.*/
/*  if (context_.exists(key)) {*/
/*    // For scalar, warn if overwriting task-level option (global substitution is allowed with no warning).*/
/*    // Logging is performed via Logger.*/
/*    tadah::core::Logger::instance().log(tadah::core::LogLevel::WARNING, */
/*      "Task ", taskID_, ": Overwriting option ", key);*/
/*  }*/
/*  // Modify the context with task-specific metadata.*/
/*  Context temp = context_;*/
/*  tadah::core::MetaData meta;*/
/*  meta.isGlobal = false;*/
/*  context_.set(key, value, meta);*/
/*}*/
/**/
/*std::string Task::getOption(const std::string &key) const {*/
/*  return context_.get(key);*/
/*}*/
/**/
/*const Context& Task::effectiveContext() const {*/
/*  return context_;*/
/*}*/
/**/
/*void Task::setPriority(int p) {*/
/*  priority_ = p;*/
/*}*/
/**/
/*int Task::getPriority() const {*/
/*  return priority_;*/
/*}*/
/**/
/*std::string Task::getID() const {*/
/*  return taskID_;*/
/*}*/
/**/
/*std::string Task::getTimestamp() const {*/
/*  return timestamp_;*/
/*}*/
/**/
/*std::string Task::serialize() const {*/
/*  // Format: taskID;command;timestamp;priority\n followed by context serialization.*/
/*  std::ostringstream oss;*/
/*  oss << taskID_ << ";" << command_ << ";" << timestamp_ << ";" << priority_ << "\n"*/
/*      << context_.serialize();*/
/*  return oss.str();*/
/*}*/
/**/
/*std::vector<char> Task::serializeBinary() const {*/
/*  // Example binary format: first the length of taskID (4 bytes) then taskID, then similarly for command and timestamp,*/
/*  // followed by priority (4 bytes). Only selected fields are serialized.*/
/*  std::vector<char> buffer;*/
/*  auto appendString = [&buffer](const std::string &s) {*/
/*    std::uint32_t len = static_cast<std::uint32_t>(s.size());*/
/*    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&len), reinterpret_cast<const char*>(&len) + sizeof(len));*/
/*    buffer.insert(buffer.end(), s.begin(), s.end());*/
/*  };*/
/*  appendString(taskID_);*/
/*  appendString(command_);*/
/*  appendString(timestamp_);*/
/*  std::uint32_t prio = static_cast<std::uint32_t>(priority_);*/
/*  buffer.insert(buffer.end(), reinterpret_cast<const char*>(&prio), reinterpret_cast<const char*>(&prio) + sizeof(prio));*/
/*  return buffer;*/
/*}*/
/**/
/*Task Task::deserialize(const std::string &data, const Context &globalContext) {*/
/*  std::istringstream iss(data);*/
/*  std::string header;*/
/*  std::getline(iss, header);*/
/*  std::istringstream headerStream(header);*/
/*  std::string id, cmd, timeStamp;*/
/*  int prio;*/
/*  if (!std::getline(headerStream, id, ';') ||*/
/*      !std::getline(headerStream, cmd, ';') ||*/
/*      !std::getline(headerStream, timeStamp, ';') ||*/
/*      !(headerStream >> prio)) {*/
/*    throw std::runtime_error("Failed to deserialize Task header");*/
/*  }*/
/*  Task t(cmd, globalContext);*/
/*  // Override fields with parsed values.*/
/*  // For simplicity, set the task fields directly.*/
/*  t.taskID_ = id;*/
/*  t.timestamp_ = timeStamp;*/
/*  t.priority_ = prio;*/
/*  // Further context deserialization is omitted for brevity.*/
/*  return t;*/
/*}*/
/**/
/*void Task::print() const {*/
/*  std::cout << *this;*/
/*}*/
/**/
/*std::ostream& operator<<(std::ostream &os, const Task &t) {*/
/*  os << "Task ID: " << t.taskID_ << "\n"*/
/*     << "Command: " << t.command_ << "\n"*/
/*     << "Timestamp: " << t.timestamp_ << "\n"*/
/*     << "Priority: " << t.priority_ << "\n"*/
/*     << "Context:\n" << t.context_;*/
/*  return os;*/
/*}*/
/**/
/*void Task::setDependency(const std::string &dependencyRule) {*/
/*  dependencyRule_ = dependencyRule;*/
/*}*/
/**/
/*std::string Task::getTaskType() const {*/
/*  // Assumes that the first word of the command defines the type.*/
/*  std::istringstream iss(command_);*/
/*  std::string type;*/
/*  iss >> type;*/
/*  return type;*/
/*}*/
/**/
/*std::string Task::generateTaskID() const {*/
/*  std::ostringstream oss;*/
/*  // Generate based on current time in nanoseconds.*/
/*  oss << std::chrono::system_clock::now().time_since_epoch().count();*/
/*  return oss.str();*/
/*}*/
/**/
/*std::string Task::generateTimestamp() const {*/
/*  auto now = std::chrono::system_clock::now();*/
/*  std::time_t now_c = std::chrono::system_clock::to_time_t(now);*/
/*  std::string ts = std::ctime(&now_c);*/
/*  if (!ts.empty() && ts.back() == '\n') ts.pop_back();*/
/*  return ts;*/
/*}*/
/**/
/*} // namespace mlip*/
/*} // namespace tadah*/
/**/
