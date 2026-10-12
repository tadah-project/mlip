// The engine registry and the scheduler: a task whose engine this program
// does not have (none registered, or registered as not built, as `tadah
// hpo` in a tadah built without TADAH_ENABLE_NFP) is refused before any
// task runs, with the message the program gave for it.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/mlip/engines/engine_factory.h>
#include <tadah/mlip/engines/iengine.h>
#include <tadah/mlip/scheduler.h>
#include <tadah/mlip/task.h>

#include <memory>
#include <stdexcept>
#include <string>

using tadah::core::Context;
using tadah::mlip::Scheduler;
using tadah::mlip::Task;
using tadah::mlip::engines::EngineFactory;
using tadah::mlip::engines::IEngine;

namespace {

// An engine that counts the runs of every instance.
struct CountingEngine : IEngine {
  static int runs;
  void validate(const Task &) override {}
  int run(Task &) override {
    ++runs;
    return 0;
  }
};
int CountingEngine::runs = 0;

// The message of the exception `f` throws, or "" when it throws none.
template <class F> std::string thrown(F &&f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

Task task(const std::string &name) { return Task(name, Context()); }

}  // namespace

/* Pins: a scheduler holding a task with an engine (TEST.factory.counting,
 * which counts its runs) followed by a task no engine is registered for
 * (TEST.factory.unknown) throws "EngineFactory::create: No engine found for
 * TEST.factory.unknown" from execute() and runs neither task: the counting
 * engine ran 0 times.
 * Before the fix: execute() created each engine when it reached the task,
 * so the counting task ran (1 run) and the refusal came after it. */
TEST_CASE("Scheduler: a task without an engine is refused before any task "
          "runs",
          "[engine_factory][scheduler]") {
  EngineFactory::registerEngine("TEST.factory.counting",
                                [] { return std::make_unique<CountingEngine>(); });
  CountingEngine::runs = 0;
  Scheduler s;
  s.addTask(task("TEST.factory.counting"));
  s.addTask(task("TEST.factory.unknown"));
  const std::string what = thrown([&] { s.execute(); });
  CHECK(what == "EngineFactory::create: No engine found for TEST.factory.unknown");
  CHECK(CountingEngine::runs == 0);
}

/* Pins: a task registered as not built (registerNotBuilt) with a message:
 * notBuiltMessage() returns that message; create() and require() throw it,
 * as written; a scheduler holding a task with an engine and then the
 * not-built task throws it from execute() and runs neither (the counting
 * engine ran 0 times). A task with an engine, or none registered, has an
 * empty notBuiltMessage(); require() of a task with an engine throws
 * nothing and builds no engine (0 runs, and execute() then runs it once).
 * registerEngine() and registerNotBuilt() replace each other: an engine
 * registered after the not-built entry is created, and a not-built entry
 * registered after an engine refuses it.
 * Before the fix: does not compile (EngineFactory has no registerNotBuilt,
 * notBuiltMessage or require). */
TEST_CASE("EngineFactory: a task registered as not built is refused with its "
          "message",
          "[engine_factory][scheduler]") {
  const std::string msg =
      "tadah test needs a part this program was built without";
  EngineFactory::registerEngine("TEST.factory.counting",
                                [] { return std::make_unique<CountingEngine>(); });
  EngineFactory::registerNotBuilt("TEST.factory.notbuilt", msg);
  CHECK(EngineFactory::notBuiltMessage("TEST.factory.notbuilt") == msg);
  CHECK(EngineFactory::notBuiltMessage("TEST.factory.counting").empty());
  CHECK(EngineFactory::notBuiltMessage("TEST.factory.unknown").empty());
  CHECK(thrown([] { EngineFactory::create("TEST.factory.notbuilt"); }) == msg);
  CHECK(thrown([] { EngineFactory::require("TEST.factory.notbuilt"); }) == msg);
  CHECK(thrown([] { EngineFactory::require("TEST.factory.unknown"); }) ==
        "EngineFactory::create: No engine found for TEST.factory.unknown");

  CountingEngine::runs = 0;
  CHECK(thrown([] { EngineFactory::require("TEST.factory.counting"); }).empty());
  CHECK(CountingEngine::runs == 0);
  {
    Scheduler s;
    s.addTask(task("TEST.factory.counting"));
    s.addTask(task("TEST.factory.notbuilt"));
    CHECK(thrown([&] { s.execute(); }) == msg);
    CHECK(CountingEngine::runs == 0);
  }
  {
    Scheduler s;
    s.addTask(task("TEST.factory.counting"));
    CHECK(s.execute() == 0);
    CHECK(CountingEngine::runs == 1);
  }

  // Each registration replaces the other.
  EngineFactory::registerEngine("TEST.factory.notbuilt",
                                [] { return std::make_unique<CountingEngine>(); });
  CHECK(EngineFactory::notBuiltMessage("TEST.factory.notbuilt").empty());
  CHECK(EngineFactory::create("TEST.factory.notbuilt") != nullptr);
  EngineFactory::registerNotBuilt("TEST.factory.notbuilt", msg);
  CHECK(thrown([] { EngineFactory::create("TEST.factory.notbuilt"); }) == msg);
}
