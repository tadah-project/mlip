#include "catch2/catch.hpp"
#include <tadah/core/memory/IWorkspaceManager.h>

namespace tadah {
namespace core {
namespace memory {

// A minimal concrete mock derived from IWorkspaceManager
class MockWorkspaceManager : public IWorkspaceManager {
public:
    // Provide a virtual destructor, automatically calls base ~IWorkspaceManager()
    ~MockWorkspaceManager() override = default;
    // Implement any pure virtual methods if the interface has them
    // For example:
    // void somePureVirtualFunction() override { /* do nothing */ }
};

} // namespace memory
} // namespace core
} // namespace tadah

TEST_CASE("IWorkspaceManager mock derived class", "[IWorkspaceManager]") 
{
    using namespace tadah::core::memory;

    // We simply create and destroy a MockWorkspaceManager object.
    // This confirms that ~IWorkspaceManager() is indeed callable and causes no errors.
    MockWorkspaceManager mgr;
    REQUIRE(true); // If we reach this line, construction & destruction work.
}

