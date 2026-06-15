# Debug target — not part of the default build.
# Include from tt-train/tests/CMakeLists.txt via:
#   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../debug/rng_tests rng_determinism_tests_build EXCLUDE_FROM_ALL)
# Then build with:
#   cmake --build build_Release --target rng_determinism_tests -j

add_executable(rng_determinism_tests test_parallel_rng_determinism.cpp)
target_link_libraries(
    rng_determinism_tests
    PRIVATE
        GTest::gmock_main
        ttml
)
set_target_properties(
    rng_determinism_tests
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            TRUE
)
target_compile_options(rng_determinism_tests PRIVATE -fPIC)
target_precompile_headers(rng_determinism_tests REUSE_FROM ttml_pch)
include(GoogleTest)
gtest_discover_tests(rng_determinism_tests)
