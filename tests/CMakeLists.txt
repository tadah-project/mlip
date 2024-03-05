Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.7
    )

FetchContent_MakeAvailable(Catch2)

file(GLOB tests_src "*.cpp")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

find_library(tadah.core NAMES tadah.core
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../CORE/*
    )
if(tadah.core)
    message(STATUS "Found ${tadah.core}") 
else()
    message(FATAL_ERROR "
    Could not find ltadah.core in the CORE module. 
    Hint: Recompile CORE.
    ")
endif()
find_library(tadah.models NAMES tadah.models
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../MODELS/*
    )
if(tadah.models)
    message(STATUS "Found ${tadah.models}") 
else()
    message(FATAL_ERROR "
    Could not find ltadah.models in the MODELS module. 
    Hint: Recompile MODELS.
    ")
endif()

add_executable(test_mlip ${tests_src})
target_link_libraries(test_mlip PRIVATE Catch2::Catch2 tadah.mlip ${tadah.core} ${tadah.models})

if(TADAH_ENABLE_OPENMP) 
    find_package(OpenMP REQUIRED)
    target_link_libraries(test_mlip PUBLIC OpenMP::OpenMP_CXX)
endif()

include(Catch.cmake)
catch_discover_tests(test_mlip)
