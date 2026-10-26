#include <tadah/mlip/scheduler.h>
#include <tadah/mlip/engines/engine_factory.h>
#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <sstream>

namespace tadah {
namespace mlip {

Scheduler::Scheduler() {}

void Scheduler::addTask(const Task &task) {
  tasks_.push_back(task);
}
void Scheduler::addTasks(std::vector<Task> tasks) {
  tasks_.insert(tasks_.end(), tasks.begin(), tasks.end());
}

void Scheduler::schedule() {
  // Sort tasks by task type (grouping) and then by priority (descending order).*/
  std::stable_sort(tasks_.begin(), tasks_.end(),
    [](const Task& a, const Task& b) {
      if (a.getTaskName() != b.getTaskName())
        return a.getTaskName() < b.getTaskName();
      return a.getPriority() > b.getPriority();
    });
}

int Scheduler::execute() {
  // Every task's engine first: a task this program has no engine for (one
  // it was built without, such as hpo in a tadah without TADAH_ENABLE_NFP)
  // is refused before an earlier task runs.
  for (const auto &task : tasks_)
    mlip::engines::EngineFactory::require(task.getTaskName());

  for (auto &task : tasks_) {

    core::Context context = task.getContext();

    // The command as it is typed: "tadah data sample" for DATA.sample.
    std::string command = "tadah " + core::to_lower(task.getTaskName());
    std::replace(command.begin(), command.end(), '.', ' ');
    core::Logger::instance().log(core::LogLevel::INFO, "Running ", command);

    auto engine = mlip::engines::EngineFactory::create(task.getTaskName());
    engine->validate(task);
    const int status = engine->run(task);

    // An engine that reports failure through its status stops the run;
    // the status reaches the process exit code through the CLI.
    if (status != 0) return status;
  }
  return 0;
}

} // namespace mlip
} // namespace tadah
