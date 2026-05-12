# Phase 2F microbench for the dPhi/dθ walker stack. Built as a
# standalone executable rather than a test — the bench prints
# parseable timing lines to stdout and is not pass/fail-gated.
#
# Run: ./bench_dphi_dtheta_walk
# Compare against bench_dphi_dtheta_walk_reference.txt for regressions.

add_executable(bench_dphi_dtheta_walk bench_dphi_dtheta_walk.cpp)
target_link_libraries(bench_dphi_dtheta_walk PRIVATE
    tadah.mlip tadah.core tadah.models)
target_include_directories(bench_dphi_dtheta_walk PRIVATE ../include)

if(TADAH_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(bench_dphi_dtheta_walk PRIVATE OpenMP::OpenMP_CXX)
endif()
