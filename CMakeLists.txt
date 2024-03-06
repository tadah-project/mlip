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

include(ExternalProject)
ExternalProject_Add(Tadah.MODELS
    GIT_REPOSITORY git@git.ecdf.ed.ac.uk:tadah/models.git
    GIT_TAG main
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MODELS
    PATCH_COMMAND  mkdir -p build
    BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MODELS/build
    INSTALL_COMMAND cmake -E echo "Skipping install step."
    )
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../MODELS/build)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../CORE/build)

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

file(GLOB MLIP_SRC CONFIGURE_DEPENDS "*.cpp" "*.h" "*.hpp")
file(GLOB_RECURSE ANALYTICS_SRC CONFIGURE_DEPENDS 
    "analytics/*.cpp" "analytics/*.h" "analytics/*.hpp")
file(GLOB_RECURSE DESIGN_MATRIX_SRC CONFIGURE_DEPENDS 
    "design_matrix/*.cpp" "design_matrix/*.h" "design_matrix/*.hpp")
file(GLOB_RECURSE MODELS_SRC CONFIGURE_DEPENDS 
    "models/*.cpp" "models/*.h" "models/*.hpp")
file(GLOB_RECURSE OUTPUT_SRC CONFIGURE_DEPENDS 
    "output/*.cpp" "output/*.h" "output/*.hpp")
#file(GLOB_RECURSE UTILS_SRC CONFIGURE_DEPENDS "utils/*.cpp")

add_library(tadah.mlip.obj OBJECT
    ${MLIP_SRC}
    ${ANALYTICS_SRC}
    ${DESIGN_MATRIX_SRC}
    ${MODELS_SRC}
    ${OUTPUT_SRC}
    )

add_dependencies(tadah.mlip.obj Tadah.MODELS)

if(TADAH_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(tadah.mlip.obj PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(tadah.mlip $<TARGET_OBJECTS:tadah.mlip.obj>)

include(CTest) 
if(TADAH_BUILD_TESTS)
    add_subdirectory(tests)
endif()

set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/catch2)
