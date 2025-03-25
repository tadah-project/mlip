cmake_minimum_required(VERSION 3.12)
project(Tadah.MLIP)

# Guard against in-source builds
# Also, make sure user does not accidently invoke cmake .. from the source dir
# of the module in case module is added as a submodule.
get_filename_component(MODULE_TEST ${CMAKE_BINARY_DIR} NAME)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR}
        OR ${MODULE_TEST} STREQUAL "MLIP"
)
  message(FATAL_ERROR "
    In-source builds not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt and CMakeFiles dir. ")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (BUILD_SHARED_LIBS)
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

  # the RPATH to be used when installing, but only if it's not a system directory
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif("${isSystemDir}" STREQUAL "-1")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Check is build supported
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" 
    OR
    CMAKE_BUILD_TYPE STREQUAL "RELEASE"
    OR
    CMAKE_BUILD_TYPE STREQUAL "PROFILE"
    OR
    CMAKE_BUILD_TYPE STREQUAL "DEVELOP"
)
  message(STATUS "${TADAH} Build type:  ${CMAKE_BUILD_TYPE}.")
else()
  message(FATAL_ERROR "
  Selected build type is not supported: ${CMAKE_BUILD_TYPE}.
  Supported build types: Release, Debug, Profile Develop"
  )
endif()

# Update the documentation string of CMAKE_BUILD_TYPE for GUIs
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release Profile."
  FORCE)

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
if (CMAKE_BUILD_TYPE STREQUAL "DEVELOP")
if(NOT DEFINED ENV{TADAH_PATH})
  message(FATAL_ERROR "TADAH_PATH system variable is undefined for the ${CMAKE_BUILD_TYPE} build")
endif()
  function(find_directory_case_insensitive target_name search_dir result_var)
    execute_process(
        COMMAND find "${search_dir}" -maxdepth 1 -type d -iname "${target_name}"
        OUTPUT_VARIABLE found_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(found_dir)
      set(${result_var} "${found_dir}" PARENT_SCOPE)
      message("${TADAH} Develop build: Found directory: ${found_dir}")
    else()
      set(${result_var} "" PARENT_SCOPE)
      message("${TADAH} Develop build: Directory not found.")
    endif()
  endfunction()
  find_directory_case_insensitive("models" "$ENV{TADAH_PATH}" DEVELOP_MODELS_DIR)
  set(FETCHCONTENT_SOURCE_DIR_TADAH.MODELS "${DEVELOP_MODELS_DIR}")
endif()


include(FetchContent)

if(DEFINED ENV{CI_COMMIT_REF_NAME})
  set(GIT_BRANCH "$ENV{CI_COMMIT_REF_NAME}")
else()
  execute_process(
    COMMAND bash -c "git status | sed -n '1s/^On branch //p'" 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "${TADAH}.MLIP Using GIT_BRANCH:  ${GIT_BRANCH}.")

FetchContent_Declare(
    Tadah.MODELS
    GIT_REPOSITORY https://git.ecdf.ed.ac.uk/tadah/models.git
    GIT_TAG origin/${GIT_BRANCH}
)

FetchContent_MakeAvailable(Tadah.MODELS)

file(GLOB MLIP_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_library(tadah.mlip ${MLIP_SRC})
target_link_libraries(tadah.mlip PRIVATE config::rc)
target_link_libraries(tadah.mlip PUBLIC tadah.models)
target_include_directories(tadah.mlip PUBLIC include)
target_include_directories(tadah.mlip PUBLIC external)
target_include_directories(tadah.mlip PUBLIC ${Tadah.MODELS_SOURCE_DIR}/include)
target_include_directories(tadah.mlip PUBLIC ${Tadah.CORE_SOURCE_DIR}/include)
target_include_directories(tadah.mlip PUBLIC ${Tadah.CORE_SOURCE_DIR}/external/toml11)




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
target_link_libraries(tadah.mlip PRIVATE nlohmann_json::nlohmann_json)

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

if(TADAH_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(tadah.mlip PUBLIC OpenMP::OpenMP_CXX)
endif()

if (TADAH_BUILD_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(tadah.mlip PUBLIC MPI::MPI_CXX)
  add_compile_definitions(TADAH_BUILD_MPI)
endif()


if(TADAH_BUILD_TESTS)
  include(CTest) 
  add_subdirectory(tests)
endif()

# Installation rules
install(TARGETS tadah.mlip
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/
    DESTINATION include
)
