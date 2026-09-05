# SPDX-License-Identifier: MPL-2.0
# -----------------------------------------------------------------------------
# Tests: golden-file comparison.  gen_golden.py (numpy reference) produces
# goldens under ${CMAKE_BINARY_DIR}/tests/golden once at CTest time, then the
# C++ tests compare against them.  No torch / model asset required.
# -----------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_Interpreter_FOUND)
    message(WARNING "Python3 not found — golden generation disabled; tests will be skipped")
    return()
endif()

set(PCNSF_GOLDEN_DIR "${CMAKE_BINARY_DIR}/tests/golden")
file(MAKE_DIRECTORY "${PCNSF_GOLDEN_DIR}")

# --- golden generator fixture ----------------------------------------------
add_test(
    NAME gen_golden
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_golden.py
            --out ${PCNSF_GOLDEN_DIR}
)
set_tests_properties(gen_golden PROPERTIES
    FIXTURES_SETUP golden
    LABELS "gen"
)

# --- C++ tests (each fixture-required golden) --------------------------------
function(pcnsf_add_test name binary runner)
    add_executable(${binary} ${runner})
    target_link_libraries(${binary} PRIVATE pc_nsf_hifigan::pc_nsf_hifigan mininsf::mininsf)
    target_compile_options(${binary} PRIVATE ${PCNSF_WARN_FLAGS})
    target_compile_definitions(${binary} PRIVATE PCNSF_GOLDEN_DIR="${PCNSF_GOLDEN_DIR}")
    add_test(NAME ${name} COMMAND ${binary})
    set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED golden LABELS "core")
endfunction()

pcnsf_add_test(t01_mel_ng      test_mel_ng        test_mel.cpp)
pcnsf_add_test(t02_mel_nvstft  test_mel_nvstft    test_mel_nvstft.cpp)
pcnsf_add_test(t03_source      test_source        test_source.cpp)
pcnsf_add_test(t04_gguf_meta   test_gguf_meta     test_gguf_meta.cpp)
pcnsf_add_test(t06_backend_policy test_backend_policy test_backend_policy.cpp)
pcnsf_add_test(t07_audio_ops test_audio_op_regressions test_audio_op_regressions.cpp)
if(PCNSF_VULKAN)
    find_package(Vulkan REQUIRED)
    add_executable(test_vulkan_device_policy test_vulkan_device_policy.cpp)
    target_include_directories(test_vulkan_device_policy PRIVATE
        "${ggml_SOURCE_DIR}/src/ggml-vulkan" ${Vulkan_INCLUDE_DIRS})
    target_compile_options(test_vulkan_device_policy PRIVATE ${PCNSF_WARN_FLAGS})
    add_test(NAME t08_vulkan_device_policy COMMAND test_vulkan_device_policy)
    set_tests_properties(t08_vulkan_device_policy PROPERTIES LABELS "core")
    add_executable(test_vulkan_convolution test_vulkan_convolution.cpp)
    target_link_libraries(test_vulkan_convolution PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
    target_compile_options(test_vulkan_convolution PRIVATE ${PCNSF_WARN_FLAGS})
    add_test(NAME t09_vulkan_convolution COMMAND test_vulkan_convolution Vulkan0)
    set_tests_properties(t09_vulkan_convolution PROPERTIES LABELS "gpu" SKIP_RETURN_CODE 77)
endif()
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch, gguf"
    RESULT_VARIABLE _pcnsf_converter_deps OUTPUT_QUIET ERROR_QUIET)
if(_pcnsf_converter_deps EQUAL 0)
    add_test(NAME converter COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_converter.py)
endif()
add_test(NAME patch_upgrade COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_patch_upgrade.py
    --cmake ${CMAKE_COMMAND} --ggml ${ggml_SOURCE_DIR} --mininsf ${mininsf_SOURCE_DIR}
    --pocketfft ${pocketfft_SOURCE_DIR} --dr-libs ${dr_libs_SOURCE_DIR})
# Metadata tests do not exercise GPU kernels.  Force CPU before backend
# probing so they also run in sandboxed/headless macOS environments where a
# Metal command queue cannot be created.
set_tests_properties(t04_gguf_meta PROPERTIES ENVIRONMENT "PCNSF_BACKEND=cpu")

# optional end-to-end vocode test driven by PCNSF_MODEL_GGUF
if(DEFINED ENV{PCNSF_MODEL_GGUF})
    file(TO_CMAKE_PATH "$ENV{PCNSF_MODEL_GGUF}" PCNSF_TEST_MODEL_GGUF)
    add_executable(test_vocode test_vocode.cpp)
    target_link_libraries(test_vocode PRIVATE pc_nsf_hifigan::pc_nsf_hifigan)
    target_compile_definitions(test_vocode PRIVATE
        PCNSF_MODEL_GGUF="${PCNSF_TEST_MODEL_GGUF}" PCNSF_GOLDEN_DIR="${PCNSF_GOLDEN_DIR}")
    add_test(NAME t05_vocode COMMAND test_vocode)
    add_test(NAME t05_vocode_f16_forced COMMAND test_vocode F16)
    add_test(NAME t05_vocode_kernel_fallback COMMAND test_vocode F32 compare-min-k)
    set_tests_properties(t05_vocode PROPERTIES
        FIXTURES_REQUIRED golden LABELS "e2e" ENVIRONMENT "PCNSF_BACKEND=cpu")
    set_tests_properties(t05_vocode_f16_forced PROPERTIES FIXTURES_REQUIRED golden LABELS "e2e"
        ENVIRONMENT "PCNSF_BACKEND=cpu;PCNSF_DIRECT_CONV=1;PCNSF_FUSE_IO=1")
    set_tests_properties(t05_vocode_kernel_fallback PROPERTIES FIXTURES_REQUIRED golden LABELS "e2e"
        ENVIRONMENT "PCNSF_BACKEND=cpu;PCNSF_DIRECT_CONV=1;PCNSF_FUSE_IO=1;PCNSF_DIRECT_MIN_K=1")
    if(TARGET vocoder_probe)
        add_test(NAME t10_vocoder_probe_cpu COMMAND vocoder_probe "${PCNSF_TEST_MODEL_GGUF}" 33 2 - varying)
        set_tests_properties(t10_vocoder_probe_cpu PROPERTIES LABELS "e2e" ENVIRONMENT "PCNSF_BACKEND=cpu")
        if(PCNSF_VULKAN)
            add_test(NAME t10_vocoder_probe_vulkan COMMAND vocoder_probe "${PCNSF_TEST_MODEL_GGUF}" 33 2 - varying)
            set_tests_properties(t10_vocoder_probe_vulkan PROPERTIES LABELS "e2e;gpu" ENVIRONMENT "PCNSF_BACKEND=auto")
        endif()
    endif()
endif()
