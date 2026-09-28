cmake_minimum_required(VERSION 3.16)
project(binary_transformer VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Binar/Ternar Transformer Engine (Phase 1) — tashqi og'ir dependency'siz
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BT_NATIVE        "Build with -march=native (faqat shu mashina uchun)" ON)
option(BT_ENABLE_VULKAN "Vulkan compute backend (agar header topilsa)"    ON)

# --- Vulkan (ixtiyoriy): faqat header kerak; runtime dlopen orqali yuklanadi
include(CheckIncludeFileCXX)
if(DEFINED BT_VULKAN_HEADER_DIR)
  include_directories(${BT_VULKAN_HEADER_DIR})
endif()
check_include_file_cxx("vulkan/vulkan.h" BT_HAVE_VULKAN_H)
if(BT_HAVE_VULKAN_H AND BT_ENABLE_VULKAN)
  set(BT_VULKAN 1)
  message(STATUS "Vulkan backend: ENABLED (headers topildi)")
else()
  set(BT_VULKAN 0)
  message(STATUS "Vulkan backend: DISABLED (vulkan/vulkan.h yo'q — stub ishlatiladi)")
endif()

# --- Shader compiler (glslc afzal, glslangValidator ham bo'ladi)
find_program(BT_GLSLC NAMES glslc glslangValidator)
set(BT_SPV ${CMAKE_BINARY_DIR}/bin/binary_matmul.spv)
if(BT_GLSLC)
  if(BT_GLSLC MATCHES "glslang")
    add_custom_command(
      OUTPUT ${BT_SPV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
      COMMAND ${BT_GLSLC} -V ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/binary_matmul.comp -o ${BT_SPV}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/binary_matmul.comp
      COMMENT "SPIR-V: binary_matmul.comp")
  else()
    add_custom_command(
      OUTPUT ${BT_SPV}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
      COMMAND ${BT_GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/binary_matmul.comp -o ${BT_SPV}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/binary_matmul.comp
      COMMENT "SPIR-V: binary_matmul.comp")
  endif()
  add_custom_target(bt_shader DEPENDS ${BT_SPV})
  message(STATUS "Shader compiler: ${BT_GLSLC}")
else()
  add_custom_target(bt_shader)
  message(STATUS "Shader compiler topilmadi — GPU benchmark uchun glslc kerak")
endif()

# --- Core kutubxona
add_library(bt_core STATIC
  src/core/ternary_simd.cpp
  src/gpu/vulkan_wrapper.cpp)
target_include_directories(bt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(bt_core PUBLIC BT_HAVE_VULKAN=${BT_VULKAN} BT_VERSION="${PROJECT_VERSION}")

find_package(Threads REQUIRED)
target_link_libraries(bt_core PUBLIC Threads::Threads)

if(BT_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" BT_HAS_MARCH)
  if(BT_HAS_MARCH)
    target_compile_options(bt_core PUBLIC -march=native)
  endif()
endif()

if(MSVC)
  target_compile_options(bt_core PRIVATE /W4)
else()
  target_compile_options(bt_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- CLI vositalar
add_executable(benchmark_engine src/cli/benchmark_engine.cpp)
target_link_libraries(benchmark_engine PRIVATE bt_core)
add_dependencies(benchmark_engine bt_shader)

add_executable(pipeline_ping src/cli/pipeline_ping.cpp)
target_link_libraries(pipeline_ping PRIVATE bt_core)

# Phase 2: distributed 70B pipeline engine
add_executable(distributed_70b_engine src/main_multi_node.cpp)
target_link_libraries(distributed_70b_engine PRIVATE bt_core)
add_dependencies(distributed_70b_engine bt_shader)

# Phase 4: 1B tezkor chat (100+ tok/s, bitta GPU)
add_executable(fast_chat_engine src/cli/fast_chat_engine.cpp)
target_link_libraries(fast_chat_engine PRIVATE bt_core)
add_dependencies(fast_chat_engine bt_shader)

# --- Self-test
enable_testing()
add_executable(bt_selftest tests/selftest.cpp)
target_link_libraries(bt_selftest PRIVATE bt_core)
add_test(NAME bt_selftest COMMAND bt_selftest)

message(STATUS "Build type     : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standart   : ${CMAKE_CXX_STANDARD}")
message(STATUS "Vulkan backend : ${BT_VULKAN}")
