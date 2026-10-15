# Catch2 and its test discovery come from cmake-tools (tadah_setup_tests()).
tadah_setup_tests()

file(GLOB tests_src CONFIGURE_DEPENDS "*.cpp")
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests_data DESTINATION .)

add_executable(test_mlip ${tests_src})
target_link_libraries(test_mlip PRIVATE
    Catch2 tadah.mlip tadah.core tadah.models)
target_include_directories(test_mlip PRIVATE ../include)

if(TADAH_ENABLE_OPENMP) 
    find_package(OpenMP REQUIRED)
    target_link_libraries(test_mlip PUBLIC OpenMP::OpenMP_CXX)
endif()

include(Catch)
catch_discover_tests(test_mlip)

# parallel_for rethrows the lowest failing index's exception; with one
# thread (the suite's setting) that index is always the first to throw, so
# a hidden case checks the rule with several threads in an entry of its own.
if(TADAH_ENABLE_OPENMP)
    add_test(NAME mlip_parallel_for_threads
        COMMAND test_mlip "[parallel_for_threads]" --warn NoTests)
    set_tests_properties(mlip_parallel_for_threads PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_THREAD_LIMIT=4"
        LABELS "mlip"
        TIMEOUT 120)
endif()

# tadah.mlip_headers (ctest -L headers): every header of include/
# compiles on its own and twice in a row (cmake-tools'
# tadah_add_header_check). Before the fix: passes as well (a pin: no MLIP
# header failed when the check was added).
tadah_add_header_check(tadah.mlip)

# tadah.mlip_text (ctest -L text): no line of the repository's text files
# carries a person's initials, a reference to co-workers, a decision
# attributed to the code's owner, the name of an AI tool, a note kept
# outside the repository or filler wording (cmake-tools'
# tadah_add_text_check). Before the fix: fails, 76 lines: 64 with a person's
# initials, 5 decision notes, 6 workspace notes and .gitignore's entry for
# an AI tool's state directory.
tadah_add_text_check(tadah.mlip)
