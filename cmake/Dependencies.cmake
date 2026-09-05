# SPDX-License-Identifier: MPL-2.0
# ----------------------------------------------------------------------------
# External dependencies — all via FetchContent, nothing vendored
# (except pocketfft_hdronly.h which the original audio_ggml vendored as a
# single header and is kept for reproducibility).
# ----------------------------------------------------------------------------
include(FetchContent)

# Don't let child projects surprise us with their own versions.
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)

# How libmininsf gets consumed:
#   D1 decision: pc-nsf-hifigan.cpp references KakaruHayate/libmininsf for the
#   mini-nsf sine source; the ggml vocoder body is this repo.

# Propagate backend toggles as ggml's own option names *before* add_subdirectory.
set(GGML_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(GGML_METAL           ${PCNSF_METAL}  CACHE BOOL "ggml: enable Metal"  FORCE)
set(GGML_CUDA            ${PCNSF_CUDA}   CACHE BOOL "ggml: enable CUDA"   FORCE)
set(GGML_VULKAN          ${PCNSF_VULKAN} CACHE BOOL "ggml: enable Vulkan" FORCE)

set(GGML_METAL_EMBED_LIBRARY ${PCNSF_METAL_EMBED_LIBRARY}
    CACHE BOOL "ggml: embed Metal shader source" FORCE)

# ggml (MIT) — tensor engine.  Pinned to v0.19.0 (matches game_ggml_cli).
# D2-revised (2026-08-30): this repo's vocoder body now *consumes* APIs added
# by KakaruHayate/ggml-audio-patch (ggml_conv_direct_1d / *_fused /
# ggml_add_leaky_relu).  "No patches applied here" was true when hifigan.cpp
# was I/O-only; it is false now.  We vendor a byte-identical snapshot of the
# 9 shipped patches into ./patches/ and have FetchContent apply them
# idempotently on every configure.  Keeping them as files (not a ggml fork)
# preserves the D2 intent: ggml remains stock upstream, the diff lives here.
#
# Keep the initial PATCH_COMMAND for clean downloads, and verify again before
# add_subdirectory on every configure (including populated and source-override trees).
set(_pcnsf_ggml_patch_dir "${CMAKE_CURRENT_SOURCE_DIR}/patches")
set(_pcnsf_ggml_patch_1 "${_pcnsf_ggml_patch_dir}/learned-ops-ggml0190.patch")
set(_pcnsf_ggml_patch_2 "${_pcnsf_ggml_patch_dir}/qvac-ops-ggml0190.patch")
set(_pcnsf_ggml_patch_3 "${_pcnsf_ggml_patch_dir}/metal-ops-ggml0190.patch")
set(_pcnsf_ggml_patch_4 "${_pcnsf_ggml_patch_dir}/vulkan-conv-direct-1d-ggml0190.patch")
set(_pcnsf_ggml_patch_5 "${_pcnsf_ggml_patch_dir}/vulkan-pipeline-cache-ggml0190.patch")
set(_pcnsf_ggml_patch_6 "${_pcnsf_ggml_patch_dir}/metal-conv-direct-1d-ggml0190.patch")
set(_pcnsf_ggml_patch_7 "${_pcnsf_ggml_patch_dir}/audio-op-fixes-ggml0190.patch")
set(_pcnsf_ggml_patch_8 "${_pcnsf_ggml_patch_dir}/cpu-direct-conv-alignment-ggml0190.patch")
set(_pcnsf_ggml_patch_9 "${_pcnsf_ggml_patch_dir}/vulkan-device-policy-ggml0190.patch")
set(_pcnsf_ggml_alias_patch "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/metal-im2col-support.patch")
set(_pcnsf_patch_args "-DGGML_PATCH_METAL_IM2COL=${_pcnsf_ggml_alias_patch}")
foreach(_i RANGE 1 9)
    list(APPEND _pcnsf_patch_args "-DGGML_PATCH_${_i}=${_pcnsf_ggml_patch_${_i}}")
endforeach()
foreach(_p IN ITEMS "${_pcnsf_ggml_patch_1}" "${_pcnsf_ggml_patch_2}" "${_pcnsf_ggml_patch_3}" "${_pcnsf_ggml_patch_4}" "${_pcnsf_ggml_patch_5}" "${_pcnsf_ggml_patch_6}" "${_pcnsf_ggml_patch_7}" "${_pcnsf_ggml_patch_8}" "${_pcnsf_ggml_patch_9}" "${_pcnsf_ggml_alias_patch}")
    if(NOT EXISTS "${_p}")
        message(FATAL_ERROR "ggml patch snapshot missing: ${_p} — sync the vendored snapshot from KakaruHayate/ggml-audio-patch (patches/)")
    endif()
endforeach()

FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggerganov/ggml.git
    GIT_TAG        v0.19.0
    GIT_SHALLOW    TRUE
    # Populate without configuring ggml until the source has been verified.
    SOURCE_SUBDIR pcnsf-deferred-configure
    PATCH_COMMAND ${CMAKE_COMMAND} -DGGML_SOURCE_DIR=<SOURCE_DIR>
                  ${_pcnsf_patch_args}
                  -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApplyGgmlPatches.cmake"
)
FetchContent_MakeAvailable(ggml)
execute_process(
    COMMAND ${CMAKE_COMMAND} "-DGGML_SOURCE_DIR=${ggml_SOURCE_DIR}"
            ${_pcnsf_patch_args}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ApplyGgmlPatches.cmake"
    RESULT_VARIABLE _pcnsf_patch_result)
if(NOT _pcnsf_patch_result EQUAL 0)
    message(FATAL_ERROR "ggml patch verification failed; refusing to configure an incomplete source tree")
endif()
if(NOT TARGET ggml)
    add_subdirectory("${ggml_SOURCE_DIR}" "${ggml_BINARY_DIR}")
endif()
unset(_pcnsf_patch_args)
unset(_pcnsf_patch_result)
unset(_pcnsf_ggml_alias_patch)
unset(_pcnsf_ggml_patch_7)
unset(_pcnsf_ggml_patch_8)
unset(_pcnsf_ggml_patch_9)
unset(_pcnsf_ggml_patch_dir)
unset(_pcnsf_ggml_patch_1)
unset(_pcnsf_ggml_patch_2)
unset(_pcnsf_ggml_patch_3)
unset(_pcnsf_ggml_patch_4)
unset(_pcnsf_ggml_patch_5)
unset(_pcnsf_ggml_patch_6)

# libmininsf (MPL-2.0) — mini-nsf sine source generator.
FetchContent_Declare(
    mininsf
    GIT_REPOSITORY https://github.com/KakaruHayate/libmininsf.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
set(MININSF_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(MININSF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mininsf)

# pocketfft (BSD-3-Clause) — header-only STFT r2c FFT used by mel.cpp.
# Idempotent: when this repo is aggregated (hifisampler/hachitune), the
# parent may already define the same helper target.
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        32424d2067c2e8043dc646a4e49754b2b40cc549   # cpp @ 2025-10
)
FetchContent_MakeAvailable(pocketfft)
if(NOT TARGET pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft SYSTEM INTERFACE "${pocketfft_SOURCE_DIR}")
endif()

# dr_libs (Public Domain/MIT-0) — single-header WAV writer used by the CLI.
FetchContent_Declare(
    dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG        243e26ffa08a24dc8ae2e7a8c57123d9e504690c   # master @ 2025-10
)
FetchContent_MakeAvailable(dr_libs)
if(NOT TARGET dr_wav)
    add_library(dr_wav INTERFACE)
    target_include_directories(dr_wav SYSTEM INTERFACE "${dr_libs_SOURCE_DIR}")
endif()

message(STATUS "Third-party fetched:")
message(STATUS "  ggml       ${ggml_SOURCE_DIR}")
message(STATUS "  mininsf    ${mininsf_SOURCE_DIR}")
message(STATUS "  pocketfft  ${pocketfft_SOURCE_DIR}")
message(STATUS "  dr_libs    ${dr_libs_SOURCE_DIR}")
