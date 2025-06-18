cmake_minimum_required(VERSION 3.12)
project(Tadah.MLIP LANGUAGES CXX)

# 0. Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR
    "In-source builds are not allowed.\n"
    "Create a separate build directory and run CMake from there.")
endif()

# 1. Global settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# default build type for single-config generators (Ninja/Make)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(BUILD_SHARED_LIBS)
  # Always full RPATH
  # use, i.e. don't skip the full RPATH for the build tree
  set(CMAKE_SKIP_BUILD_RPATH FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# 2. Bring in the shared compile/link options
include(FetchContent)
if(NOT TARGET tadah::compile_options)
  FetchContent_Declare (
    tadah_cmake
    GIT_REPOSITORY  https://git.ecdf.ed.ac.uk/tadah/cmake-tools.git
    GIT_TAG         main
  )
  FetchContent_MakeAvailable(tadah_cmake)   # defines tadah::compile_options
endif()

# (option) Developer builds -DTADAH_DEVELOP=ON
tadah_use_local_module(Tadah.MODELS MODELS)  # FetchContent name + folder name

# 2. Bring in the shared compile/link options
include(FetchContent)
if(NOT TARGET tadah::compile_options)
  FetchContent_Declare (
    tadah_cmake
    GIT_REPOSITORY  https://git.ecdf.ed.ac.uk/tadah/cmake-tools.git
    GIT_TAG         main
  )
  FetchContent_MakeAvailable(tadah_cmake)   # defines tadah::compile_options
endif()

# 3. Git branch information, for informational purposes 
tadah_get_git_branch(GIT_BRANCH)
message(STATUS "${PROJECT_NAME} module is using GIT_BRANCH: ${GIT_BRANCH}")

#4. Fetch Tadah! modules
FetchContent_Declare(
    Tadah.MODELS
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/models.git
    GIT_TAG origin/${GIT_BRANCH}
)
FetchContent_MakeAvailable(Tadah.MODELS)

file(GLOB MLIP_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_library(tadah.mlip ${MLIP_SRC})
target_include_directories(tadah.mlip
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external>
)
target_link_libraries(tadah.mlip
  PUBLIC
    tadah::compile_options    # get warnings/asan/-pg flags
    tadah.models)

# Hack to skip the Gemmi cpptest
# 1) Temporarily store the original add_test() so we can call it.
#    We name it "_original_add_test" internally.
#
#    NOTE: We must first verify that add_test is indeed available
#    as a command to avoid scope issues.
if(COMMAND add_test)
  cmake_policy(PUSH)

  # Create an internal function that calls the real add_test().
  # By default, "add_test" is a built-in command in CMake. We are
  # giving it a new name here: _original_add_test.
  function(_original_add_test)
    # Call the built-in add_test with all arguments passed in
    _add_test(${ARGV})
  endfunction()

  # 2) Now we redefine add_test() so that it checks if the name is "cpptest".
  #    If so, we skip registering the test. Otherwise, we hand off to original_add_test.
  function(add_test)
    # Common usage is: add_test(NAME <testname> COMMAND <exe> ...)
    # So ARGV0 is "NAME", ARGV1 is the actual test name, etc.
    if(NOT "${ARGV0}" STREQUAL "NAME" OR NOT "${ARGV1}" STREQUAL "cpptest")
      # If it’s not "cpptest," just call the original
      _original_add_test(${ARGV})
    else()
      message(STATUS "Skipping unwanted Gemmi cpptest.")
    endif()
  endfunction()

  cmake_policy(POP)
endif()

###
# Now fetch or add Gemmi in the usual way. For example:
###
message(STATUS "Fetching Gemmi...")
FetchContent_Declare(
  gemmi
  GIT_REPOSITORY https://github.com/project-gemmi/gemmi.git
  GIT_TAG        v0.7.0
  GIT_SHALLOW 1
  GIT_CONFIG core.progress=true
  LOG_DOWNLOAD ON
)
FetchContent_MakeAvailable(gemmi)

target_link_libraries(tadah.mlip PRIVATE gemmi::headers)

message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW 1
  GIT_CONFIG core.progress=true
  LOG_DOWNLOAD ON
)
FetchContent_MakeAvailable(json)
target_link_libraries(tadah.mlip PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)

if (NOT TADAH_BUILD_DOCS_ONLY)
  find_package(CURL QUIET)   # Use QUIET to avoid CMake’s default error

  if(NOT CURL_FOUND)
    message(FATAL_ERROR
      "\nCould not find libcurl. Please install the development files.\n"
      "On Ubuntu/Debian-based systems, try:\n"
      "  sudo apt-get install libcurl4-openssl-dev\n"
      "On Fedora/Red Hat, try:\n"
      "  sudo dnf install libcurl-devel\n"
      "Or specify CURL_INCLUDE_DIR/CURL_LIBRARY manually.\n"
    )
  endif()

  target_link_libraries(tadah.mlip PRIVATE CURL::libcurl)
endif()

FetchContent_Declare(
  nlopt
  GIT_REPOSITORY https://github.com/stevengj/nlopt.git
  GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(nlopt)
target_link_libraries(tadah.mlip PRIVATE nlopt)

if(TADAH_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(tadah.mlip PUBLIC OpenMP::OpenMP_CXX)
endif()

if (TADAH_BUILD_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(tadah.mlip PUBLIC MPI::MPI_CXX)
  add_compile_definitions(TADAH_BUILD_MPI)
endif()

set(BUILD_TESTING OFF CACHE BOOL "Build tests")
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# Installation rules
install(TARGETS tadah.mlip
        EXPORT  tadahMlipTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)
install(EXPORT tadahMlipTargets
        NAMESPACE tadah::
        DESTINATION lib/cmake/tadah.mlip)
