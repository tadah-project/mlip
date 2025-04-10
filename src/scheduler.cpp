/*#include <tadah/mlip/scheduler.h>*/
/*#include <algorithm>*/
/*#include <iostream>*/
/*#include <sstream>*/
/**/
/*namespace tadah {*/
/*namespace mlip {*/
/**/
/*Scheduler::Scheduler() {}*/
/**/
/*void Scheduler::addTask(const std::shared_ptr<Task> &task) {*/
/*  tasks_.push_back(task);*/
/*}*/
/**/
/*void Scheduler::schedule() {*/
/*  // Sort tasks by task type (grouping) and then by priority (descending order).*/
/*  std::stable_sort(tasks_.begin(), tasks_.end(),*/
/*    [](const std::shared_ptr<Task>& a, const std::shared_ptr<Task>& b) {*/
/*      if (a->getTaskName() != b->getTaskName())*/
/*        return a->getTaskName() < b->getTaskName();*/
/*      return a->getPriority() > b->getPriority();*/
/*    });*/
/*  // Stub: if dependency rules are set, adjust ordering accordingly.*/
/*}*/
/**/
/*void Scheduler::execute() {*/
/*  for (const auto &task : tasks_) {*/
/*    task->print();*/
/*    std::cout << "---- Executing Task: " << task->getID() */
/*              << " (" << task->getTaskName() << ")\n";*/
/*  }*/
/*}*/
/**/
/*void Scheduler::setDependencyRule(const std::string &rule) {*/
/*  dependencyRule_ = rule;*/
/*}*/
/**/
/*std::string Scheduler::serializeTasks() const {*/
/*  std::ostringstream oss;*/
/*  for (const auto &task : tasks_) {*/
/*    oss << task->serialize() << "\n";*/
/*  }*/
/*  return oss.str();*/
/*}*/
/**/
/*} // namespace mlip*/
/*} // namespace tadah*/
/**/
