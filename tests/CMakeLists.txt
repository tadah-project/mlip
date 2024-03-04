Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.7
    )

FetchContent_MakeAvailable(Catch2)

file(GLOB tests_src "*.cpp")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_mlip ${tests_src})
target_link_libraries(test_mlip PRIVATE Catch2::Catch2 tadah.mlip)

if(TADAH_ENABLE_OPENMP) 
    find_package(OpenMP REQUIRED)
    target_link_libraries(test_mlip PUBLIC OpenMP::OpenMP_CXX)
endif()

include(Catch.cmake)
catch_discover_tests(test_mlip)
