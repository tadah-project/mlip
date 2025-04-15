#include <tadah/mlip/scheduler.h>
#include <tadah/mlip/engines/engine_factory.h>
#include <algorithm>
#include <iostream>
#include <sstream>

#ifdef TADAH_BUILD_MPI
#include <mpi.h>
#endif

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

void Scheduler::distributeTasks() {
#ifdef TADAH_BUILD_MPI
  int rank=0, nprocs=1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // If only 1 process, do nothing
  if (nprocs < 2) return;

  // On rank 0, we have tasks_. On other ranks, we will reconstruct it.
  int numTasks = 0;
  if (rank == 0) {
    numTasks = static_cast<int>(tasks_.size());
  }

  // Broadcast how many tasks to expect
  MPI_Bcast(&numTasks, 1, MPI_INT, 0, MPI_COMM_WORLD);

  for (int i = 0; i < numTasks; ++i) {
    // Rank 0 serializes the i-th Task
    std::vector<char> buffer;
    if (rank == 0) {
      buffer = tasks_[i].serializeBinary(); 
    }

    // Broadcast the buffer size
    int bufSize = static_cast<int>(buffer.size());
    MPI_Bcast(&bufSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize on non-root ranks
    buffer.resize(bufSize);

    // Broadcast the actual bytes
    if (bufSize > 0) {
      MPI_Bcast(buffer.data(), bufSize, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // Reconstruct the Task on each rank except rank 0
    if (rank != 0) {
      tasks_.push_back(Task());
      tasks_.back().deserializeBinary(buffer);
    }
  }
#endif
}

void Scheduler::execute() {
  int rank=0;
#ifdef TADAH_BUILD_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  std::cout << "Rank " << rank << " executing tasks..." << std::endl;
  std::cout << "Total tasks: " << tasks_.size() << std::endl;
  for (auto &task : tasks_) {
    std::cout << "Rank " << rank << " executing task: " << task.getID() << std::endl;
    core::Context context = task.getContext();
    std::cout << "Task context: " << context << std::endl;
    task.print();

    // Possibly adjust log level from context
    size_t verbose = 0;
    if (context.contains("VERBOSE")) {
      verbose = task.getContext().get<size_t>("VERBOSE");
    }
    core::Logger::instance().setLogLevel(static_cast<core::LogLevel>(verbose));

    auto engine = mlip::engines::EngineFactory::create(task.getTaskName());
    engine->validate(task);
    engine->run(task);

#ifdef TADAH_BUILD_MPI
    // Barrier per task (should not be required, but for safety)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

} // namespace mlip
} // namespace tadah
