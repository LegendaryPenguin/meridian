cmake_minimum_required(VERSION 3.20)
project(meridian CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  # A retrieval benchmark at -O0 measures nothing: posting decode and distance
  # computation are both tight loops, and an unoptimized build is the easiest way to
  # publish a latency number that is really just the optimizer being switched off.
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# -DNDEBUG matters: without it, assert() survives into this build, and
# Batch::SetSelection asserts std::is_sorted over the whole selection vector on
# every filter. Measured at no detectable cost on the current kernel, but a
# release build that silently runs O(n) assertions per batch is a trap waiting
# for whoever optimizes the kernel.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

option(MERIDIAN_ASAN "Build with address and undefined-behaviour sanitizers" OFF)
if(MERIDIAN_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# A separate build rather than another flag on the ASAN one, because clang refuses to combine
# them -- they instrument the same loads and stores incompatibly. Keeping them apart also keeps
# what each one proves apart: ASAN covers the storage format's bounds and lifetimes, TSAN covers
# the generation-publication and cache-sharding story, and a single mixed run that failed would
# not say which.
#
# This existed only as a hand-assembled compiler invocation while the concurrency work was being
# done, which meant the project's TSAN coverage was real but not reproducible by anyone reading
# it. An option that is not in the build system is a claim nobody else can check.
option(MERIDIAN_TSAN "Build with the thread sanitizer" OFF)
if(MERIDIAN_TSAN)
  if(MERIDIAN_ASAN)
    message(FATAL_ERROR "MERIDIAN_ASAN and MERIDIAN_TSAN cannot be combined; use separate build directories")
  endif()
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
file(GLOB_RECURSE MERIDIAN_SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_library(meridian STATIC ${MERIDIAN_SOURCES})
target_include_directories(meridian PUBLIC include)

# ---------------------------------------------------------------------------
# Tools: one executable per tools/*.cpp
# ---------------------------------------------------------------------------
file(GLOB MERIDIAN_TOOLS CONFIGURE_DEPENDS tools/*.cpp)
foreach(tool_src ${MERIDIAN_TOOLS})
  get_filename_component(tool_name ${tool_src} NAME_WE)
  add_executable(${tool_name} ${tool_src})
  target_link_libraries(${tool_name} PRIVATE meridian)
  target_include_directories(${tool_name} PRIVATE tests)
endforeach()

# ---------------------------------------------------------------------------
# Tests. One executable per file keeps a crash in one suite from hiding the rest.
# ---------------------------------------------------------------------------
enable_testing()

file(GLOB MERIDIAN_TEST_SOURCES CONFIGURE_DEPENDS tests/*_test.cpp)
foreach(test_src ${MERIDIAN_TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE meridian)
  target_include_directories(${test_name} PRIVATE tests)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# The differential suite compares Quarry against SQLite, so that one test target
# links libsqlite3. Nothing else in the project depends on it.
if(TARGET differential_test)
  find_path(SQLITE3_INCLUDE_DIR sqlite3.h)
  find_library(SQLITE3_LIBRARY sqlite3)
  if(SQLITE3_INCLUDE_DIR AND SQLITE3_LIBRARY)
    target_include_directories(differential_test PRIVATE ${SQLITE3_INCLUDE_DIR})
    target_link_libraries(differential_test PRIVATE ${SQLITE3_LIBRARY})
    target_compile_definitions(differential_test PRIVATE MERIDIAN_HAVE_SQLITE3)
  else()
    message(WARNING "sqlite3 not found: differential_test will fail rather than silently skip")
  endif()
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
file(GLOB MERIDIAN_BENCH_SOURCES CONFIGURE_DEPENDS bench/*_bench.cpp)
foreach(bench_src ${MERIDIAN_BENCH_SOURCES})
  get_filename_component(bench_name ${bench_src} NAME_WE)
  add_executable(${bench_name} ${bench_src})
  target_link_libraries(${bench_name} PRIVATE meridian)
  target_include_directories(${bench_name} PRIVATE tests bench)
endforeach()
