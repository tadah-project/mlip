
file(GLOB tests_src "*.cpp")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_mlip ${tests_src})
target_link_libraries(test_mlip PRIVATE
    Catch2 tadah.mlip tadah.core tadah.models)
target_include_directories(test_mlip PRIVATE ../include)

if(TADAH_ENABLE_OPENMP) 
    find_package(OpenMP REQUIRED)
    target_link_libraries(test_mlip PUBLIC OpenMP::OpenMP_CXX)
endif()

include(Catch.cmake)
catch_discover_tests(test_mlip)
