"""Reconfigure a populated five-patch FetchContent build and check the upgrade."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(command, cwd=None):
    """Capture a command, including diagnostic output on failure."""
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, errors="replace")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def main():
    """Simulate old stamps, then validate upgrade and idempotent reconfiguration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ggml", required=True)
    parser.add_argument("--mininsf", required=True)
    parser.add_argument("--pocketfft", required=True)
    parser.add_argument("--dr-libs", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    names = ["learned-ops", "qvac-ops", "metal-ops", "vulkan-conv-direct-1d", "vulkan-pipeline-cache"]
    with tempfile.TemporaryDirectory(prefix="pcnsf-upgrade-") as directory:
        root = Path(directory)
        stock = root / "stock"
        run(["git", "clone", "--quiet", "--no-hardlinks", args.ggml, str(stock)])
        run(["git", "checkout", "--quiet", "30bf868"], stock)
        shutil.copytree(repo / "patches", root / "patches")
        shutil.copytree(repo / "cmake", root / "cmake")
        common = '''cmake_minimum_required(VERSION 3.18)
project(patch_upgrade LANGUAGES C CXX)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(PCNSF_METAL OFF)
include(FetchContent)
'''
        patch_args = "\n".join(
            f'"-DGGML_PATCH_{i+1}=${{CMAKE_CURRENT_SOURCE_DIR}}/patches/{name}-ggml0190.patch"'
            for i, name in enumerate(names)
        )
        cmakelists = root / "CMakeLists.txt"
        cmakelists.write_text(common + f'''
FetchContent_Declare(ggml GIT_REPOSITORY "{stock.as_posix()}" GIT_TAG 30bf868
    PATCH_COMMAND ${{CMAKE_COMMAND}} -DGGML_SOURCE_DIR=<SOURCE_DIR>
    {patch_args}
    -P "${{CMAKE_CURRENT_SOURCE_DIR}}/cmake/ApplyGgmlPatches.cmake")
FetchContent_MakeAvailable(ggml)
''')
        configure = [args.cmake, "-S", str(root), "-B", str(root / "build"),
                     "-DFETCHCONTENT_UPDATES_DISCONNECTED=ON"]
        run(configure)
        source = root / "build/_deps/ggml-src"
        stamps = source / ".pcnsf-patches"
        assert (stamps / "GGML_PATCH_5.stamp").exists()
        assert not (stamps / "GGML_PATCH_6.stamp").exists()
        # Simulate builds predating the Metal IM2COL_FAST_1D support fix too.
        device = source / "src/ggml-metal/ggml-metal-device.m"
        device.write_text(device.read_text(encoding="utf-8").replace("        case GGML_OP_IM2COL_FAST_1D:\n", ""), encoding="utf-8")
        cmakelists.write_text(common + '\ninclude(cmake/Dependencies.cmake)\n')
        deps = root / "cmake/Dependencies.cmake"
        deps.write_text(deps.read_text(encoding="utf-8").replace("https://github.com/ggerganov/ggml.git", stock.as_posix()), encoding="utf-8")
        configure += [f"-DFETCHCONTENT_SOURCE_DIR_MININSF={args.mininsf}",
                      f"-DFETCHCONTENT_SOURCE_DIR_POCKETFFT={args.pocketfft}",
                      f"-DFETCHCONTENT_SOURCE_DIR_DR_LIBS={args.dr_libs}"]
        run(configure)
        for patch in ["6", "7", "8", "9", "METAL_IM2COL"]:
            assert (stamps / f"GGML_PATCH_{patch}.stamp").exists(), patch
        assert "case GGML_OP_IM2COL_FAST_1D:" in device.read_text(encoding="utf-8")
        assert "kernel_conv_direct_1d_f32_64x64" in (source / "src/ggml-metal/ggml-metal.metal").read_text(encoding="utf-8")
        assert "int64_t scatter_index = idx" in (source / "src/ggml-cpu/ops.cpp").read_text(encoding="utf-8")
        vulkan = source / "src/ggml-vulkan"
        assert "ggml_vk_select_conv_direct_variant" in (vulkan / "ggml-vulkan.cpp").read_text(encoding="utf-8")
        assert "ggml_vk_memory_type_indices" in (vulkan / "ggml-vulkan-device-policy.hpp").read_text(encoding="utf-8")
        assert "0 applied" in run(configure)
        print("Populated five-patch build upgraded; second configure is idempotent")


if __name__ == "__main__":
    main()
