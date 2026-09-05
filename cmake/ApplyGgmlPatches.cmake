# SPDX-License-Identifier: MPL-2.0
# ApplyGgmlPatches.cmake — idempotent apply of the ggml-audio-patch snapshot
# onto the freshly-fetched stock ggml v0.19.0 tree.  Invoked as FetchContent
# PATCH_COMMAND:
#   cmake -DGGML_SOURCE_DIR=<dir> -DGGML_PATCH_1=.. [-DGGML_PATCH_2=.. ...] -P this.cmake
#
# Why NOT `git apply -R --check` for idempotency: the patches overlap
# contextually (e.g. the op-name table in src/ggml.c grows with each patch,
# and patches 4/5 both edit src/ggml-vulkan/ggml-vulkan.cpp).
# After applying 1..8 in order, `git apply -R --check patch_1` fails because
# patch_2's additions sit in patch_1's context lines.  Reverse order would
# work only as a strictly nested unwind — too fragile for a guard check.
#
# Scheme instead (per patch):
#   stamp file present + content marker present  → already applied → skip
#   stamp file present + marker ABSENT           → FATAL (stamp lies; refuse)
#   no stamp, forward --check OK                 → git apply → write stamp
#   no stamp, forward fails, marker PRESENT      → pre-patched tree (local
#                                                  hand-applied snapshot):
#                                                  adopt + write stamp
#   no stamp, forward fails, marker absent       → FATAL (neither stock nor
#                                                  patched: half-state)
#
# Marker = a string that exists in the tree *only* if that patch was applied.
# Markers live in include/ggml.h for patches 1/2 (API decls), in files created
# by patches 3/4 for the Vulkan/Metal ops, and (patch 5) in the env-var name
# added to src/ggml-vulkan/ggml-vulkan.cpp.

if(NOT DEFINED GGML_SOURCE_DIR)
    message(FATAL_ERROR "GGML_SOURCE_DIR not set")
endif()

find_program(GIT_EXECUTABLE git REQUIRED)

# (patch_file, marker_file, marker_regex) triples
set(_specs
    "GGML_PATCH_1|include/ggml.h|ggml_conv_direct_1d_fused"
    "GGML_PATCH_2|include/ggml.h|GGML_OP_ADD_LEAKY_RELU"
    "GGML_PATCH_3|src/ggml-metal/ggml-metal-device.cpp|kernel_supertonic_pw2_residual"
    "GGML_PATCH_4|src/ggml-vulkan/vulkan-shaders/conv_direct_1d.comp|XS_ROWS"
    "GGML_PATCH_5|src/ggml-vulkan/ggml-vulkan.cpp|GGML_VK_PIPELINE_CACHE_PATH"
    "GGML_PATCH_METAL_IM2COL|src/ggml-metal/ggml-metal-device.m|case GGML_OP_IM2COL_FAST_1D:"
    "GGML_PATCH_6|src/ggml-metal/ggml-metal.metal|kernel_conv_direct_1d_f32_64x64"
    "GGML_PATCH_7|src/ggml-cpu/ops.cpp|int64_t scatter_index = idx"
    "GGML_PATCH_8|src/ggml-cpu/ops.cpp|_mm256_loadu_ps\\(wp\\)"
    "GGML_PATCH_9|src/ggml-vulkan/ggml-vulkan.cpp|ggml_vk_select_conv_direct_variant")

set(_n_applied 0)
set(_n_skipped 0)

foreach(_spec IN LISTS _specs)
    string(REPLACE "|" ";" _parts "${_spec}")
    list(GET _parts 0 _var)
    list(GET _parts 1 _marker_file)
    list(GET _parts 2 _marker_regex)

    set(_patch "${${_var}}")
    if(NOT _patch)
        continue()
    endif()
    if(NOT EXISTS "${_patch}")
        message(FATAL_ERROR "[ggml-patch] patch file missing: ${_patch}")
    endif()

    set(_stamp "${GGML_SOURCE_DIR}/.pcnsf-patches/${_var}.stamp")
    set(_mfile "${GGML_SOURCE_DIR}/${_marker_file}")

    set(_marker_present FALSE)
    if(EXISTS "${_mfile}")
        file(READ "${_mfile}" _mcontent)
        if(_mcontent MATCHES "${_marker_regex}")
            set(_marker_present TRUE)
        endif()
    endif()

    if(EXISTS "${_stamp}")
        if(_marker_present)
            message(STATUS "[ggml-patch] already applied (stamp+marker), skip: ${_patch}")
            math(EXPR _n_skipped "${_n_skipped}+1")
            continue()
        else()
            message(FATAL_ERROR
                "[ggml-patch] stamp exists but content marker missing — the stamp\n"
                "lies about tree state. Refusing to continue.\n"
                "  stamp: ${_stamp}\n  marker: ${_marker_file} !~ ${_marker_regex}")
        endif()
    endif()

    # No stamp: try forward apply.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check "${_patch}"
        WORKING_DIRECTORY "${GGML_SOURCE_DIR}"
        RESULT_VARIABLE _rc_check OUTPUT_QUIET ERROR_QUIET)
    if(_rc_check EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" apply "${_patch}"
            WORKING_DIRECTORY "${GGML_SOURCE_DIR}"
            RESULT_VARIABLE _rc_apply OUTPUT_QUIET ERROR_QUIET)
        if(NOT _rc_apply EQUAL 0)
            message(FATAL_ERROR "[ggml-patch] git apply failed after check passed (race?): ${_patch}")
        endif()
        file(MAKE_DIRECTORY "${GGML_SOURCE_DIR}/.pcnsf-patches")
        file(WRITE "${_stamp}" "applied-by: ApplyGgmlPatches.cmake\npatch: ${_patch}\n")
        message(STATUS "[ggml-patch] applied: ${_patch}")
        math(EXPR _n_applied "${_n_applied}+1")
    else()
        if(_marker_present)
            # Tree was hand-patched without our stamp (dev box).  Adopt it.
            file(MAKE_DIRECTORY "${GGML_SOURCE_DIR}/.pcnsf-patches")
            file(WRITE "${_stamp}" "adopted-preexisting: ApplyGgmlPatches.cmake\npatch: ${_patch}\n")
            message(STATUS "[ggml-patch] pre-patched tree detected, adopted: ${_patch}")
            math(EXPR _n_skipped "${_n_skipped}+1")
        else()
            message(FATAL_ERROR
                "[ggml-patch] cannot reconcile tree state with patch:\n"
                "  patch : ${_patch}\n"
                "  source: ${GGML_SOURCE_DIR}\n"
                "Forward check fails and no content marker found — the tree is\n"
                "neither stock v0.19.0 nor pre-patched.  Inspect manually; do\n"
                "NOT bypass: a half-applied tree produces silently wrong kernels.")
        endif()
    endif()
endforeach()

message(STATUS "[ggml-patch] resolved: ${_n_applied} applied, ${_n_skipped} already present/adopted")
