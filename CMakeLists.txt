# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.18)
project(pc_nsf_hifigan
    VERSION 0.1.0
    DESCRIPTION "NSF-HiFiGAN (mini_nsf) native ggml inference engine"
    LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(PCNSF_CUDA  "Enable ggml CUDA backend"  OFF)
option(PCNSF_VULKAN "Enable ggml Vulkan backend" OFF)
option(PCNSF_BUILD_CLI    "Build hifigan_cli executable" ON)
option(PCNSF_BUILD_TESTS  "Build C++ tests"              ON)
option(PCNSF_BUILD_EXAMPLES "Build external consumer example" OFF)
option(PCNSF_BUILD_BENCHMARK "Build pcnsf_benchmark executable" OFF)

# On Apple, default to Metal backend on. Use PCNSF_METAL to override.
if(APPLE)
    option(PCNSF_METAL "Enable ggml Metal backend" ON)
    option(PCNSF_METAL_EMBED_LIBRARY
           "Embed Metal shader source (no separately installed Metal Toolchain required)" ON)
else()
    option(PCNSF_METAL "Enable ggml Metal backend" OFF)
    option(PCNSF_METAL_EMBED_LIBRARY "Embed Metal shader source" OFF)
endif()

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
if(MSVC)
    foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
        string(TOUPPER "${_cfg}" _CFG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/bin")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_CFG} "${CMAKE_BINARY_DIR}/lib")
    endforeach()
    # MSVC-only flag; must NOT reach NVCC (a bare /utf-8 token makes nvcc
    # treat it as an extra input file -> "A single input file is required").
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

if(MSVC)
    set(PCNSF_WARN_FLAGS /W3 /wd4244 /wd4267 /wd4996)
else()
    set(PCNSF_WARN_FLAGS -Wall -Wextra -Wno-unused-parameter)
endif()

# -----------------------------------------------------------------------------
# Dependencies (all via FetchContent)
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Dependencies.cmake)

# -----------------------------------------------------------------------------
# Core library: pc_nsf_hifigan
# -----------------------------------------------------------------------------
set(PCNSF_CORE_SOURCES
    src/gguf_model.cpp
    src/mel.cpp
    src/hifigan.cpp
)

add_library(pc_nsf_hifigan STATIC ${PCNSF_CORE_SOURCES})
add_library(pc_nsf_hifigan::pc_nsf_hifigan ALIAS pc_nsf_hifigan)

target_include_directories(pc_nsf_hifigan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(pc_nsf_hifigan
    PUBLIC  ggml mininsf::mininsf
)
target_link_libraries(pc_nsf_hifigan PRIVATE pocketfft)
if(MINGW AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    # gcc<9 needs the separate std::filesystem runtime on MinGW
    target_link_libraries(pc_nsf_hifigan PUBLIC stdc++fs)
endif()

target_compile_options(pc_nsf_hifigan PRIVATE ${PCNSF_WARN_FLAGS})
target_compile_definitions(pc_nsf_hifigan PUBLIC
    PCNSF_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    PCNSF_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    PCNSF_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_CLI)
    add_executable(hifigan_cli tools/hifigan_cli.cpp)
    target_link_libraries(hifigan_cli PRIVATE pc_nsf_hifigan::pc_nsf_hifigan dr_wav)
    target_compile_options(hifigan_cli PRIVATE ${PCNSF_WARN_FLAGS})
    add_executable(test_conv_direct tools/test_conv_direct.cpp)
    target_link_libraries(test_conv_direct PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
    target_compile_options(test_conv_direct PRIVATE ${PCNSF_WARN_FLAGS})
    add_executable(vocoder_probe tools/vocoder_probe.cpp)
    target_link_libraries(vocoder_probe PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
    target_compile_options(vocoder_probe PRIVATE ${PCNSF_WARN_FLAGS})
endif()

# -----------------------------------------------------------------------------
# Benchmark (ggml vs ONNX Runtime CPU)
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_BENCHMARK)
    if(NOT ONNX_RUNTIME_ROOT)
        set(ONNX_RUNTIME_ROOT "$ENV{ONNX_RUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")
    endif()
    if(NOT ONNX_RUNTIME_ROOT)
        message(FATAL_ERROR "PCNSF_BUILD_BENCHMARK=ON requires ONNX_RUNTIME_ROOT to be set")
    endif()

    # ONNX_RUNTIME_ROOT must point at an *official* ONNX Runtime release package
    # (the layout produced by onnxruntime-{win-x64,linux-x64,osx-arm64}-X.Y.Z):
    #   include/onnxruntime_c_api.h   <- flat public C API header
    #   lib/onnxruntime.lib | lib/libonnxruntime.{so,dylib}
    # No headers or import libraries are vendored in this repository; the SDK is
    # always fetched (see .github/workflows/benchmark.yml).
    set(ORT_INC "${ONNX_RUNTIME_ROOT}/include")
    if(NOT EXISTS "${ORT_INC}/onnxruntime_c_api.h")
        message(FATAL_ERROR
            "onnxruntime_c_api.h not found under ${ORT_INC}. "
            "ONNX_RUNTIME_ROOT must point at an extracted official ONNX Runtime release package.")
    endif()

    if(WIN32)
        set(ORT_LIB_CANDIDATES "${ONNX_RUNTIME_ROOT}/lib/onnxruntime.lib"
                               "${ONNX_RUNTIME_ROOT}/onnxruntime.lib")
    elseif(APPLE)
        set(ORT_LIB_CANDIDATES "${ONNX_RUNTIME_ROOT}/lib/libonnxruntime.dylib"
                               "${ONNX_RUNTIME_ROOT}/libonnxruntime.dylib")
    else()
        set(ORT_LIB_CANDIDATES "${ONNX_RUNTIME_ROOT}/lib/libonnxruntime.so"
                               "${ONNX_RUNTIME_ROOT}/libonnxruntime.so")
    endif()

    set(ORT_LIB "")
    foreach(cand IN LISTS ORT_LIB_CANDIDATES)
        if(EXISTS "${cand}")
            set(ORT_LIB "${cand}")
            break()
        endif()
    endforeach()
    if(NOT ORT_LIB)
        message(FATAL_ERROR
            "ONNX Runtime link library not found. Looked for: ${ORT_LIB_CANDIDATES}")
    endif()
    message(STATUS "  ONNX Runtime ....... ${ORT_LIB}")

    if(WIN32)
        # Runtime DLLs live in bin/ in the release zip; older layouts keep them at the root.
        file(GLOB ORT_RUNTIME_DLLS
            "${ONNX_RUNTIME_ROOT}/lib/onnxruntime*.dll"
            "${ONNX_RUNTIME_ROOT}/bin/onnxruntime*.dll"
            "${ONNX_RUNTIME_ROOT}/onnxruntime*.dll")
    elseif(APPLE)
        file(GLOB ORT_RUNTIME_DLLS "${ONNX_RUNTIME_ROOT}/lib/libonnxruntime*.dylib")
    else()
        file(GLOB ORT_RUNTIME_DLLS "${ONNX_RUNTIME_ROOT}/lib/libonnxruntime.so*")
    endif()

    add_executable(pcnsf_benchmark
        tools/benchmark/main.cpp
        tools/benchmark/cpu_info.cpp
        tools/benchmark/timer.cpp
        tools/benchmark/ggml_runner.cpp
        tools/benchmark/ort_runner.cpp
        tools/benchmark/accuracy.cpp
        tools/benchmark/report.cpp
    )
    target_include_directories(pcnsf_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/benchmark
        ${ORT_INC}
    )
    target_link_libraries(pcnsf_benchmark PRIVATE
        pc_nsf_hifigan::pc_nsf_hifigan
        ${ORT_LIB}
    )
    target_compile_options(pcnsf_benchmark PRIVATE ${PCNSF_WARN_FLAGS})

    # Copy the ONNX Runtime shared libraries next to the executable so the
    # packaged benchmark runs without any extra environment setup.
    foreach(dll IN LISTS ORT_RUNTIME_DLLS)
        add_custom_command(TARGET pcnsf_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}" "$<TARGET_FILE_DIR:pcnsf_benchmark>")
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
if(PCNSF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(DIRECTORY include/pc_nsf_hifigan
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(TARGETS pc_nsf_hifigan
        EXPORT pc_nsf_hifigan-targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

message(STATUS "")
message(STATUS "pc_nsf_hifigan ${PROJECT_VERSION} configuration:")
message(STATUS "  Metal .............. ${PCNSF_METAL}")
message(STATUS "  Metal shader embed . ${PCNSF_METAL_EMBED_LIBRARY}")
message(STATUS "  CUDA ............... ${PCNSF_CUDA}")
message(STATUS "  Vulkan ............. ${PCNSF_VULKAN}")
message(STATUS "  Build CLI .......... ${PCNSF_BUILD_CLI}")
message(STATUS "  Build Tests ........ ${PCNSF_BUILD_TESTS}")
message(STATUS "  Benchmark .......... ${PCNSF_BUILD_BENCHMARK}")
message(STATUS "")
