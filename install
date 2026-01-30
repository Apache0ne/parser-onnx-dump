import argparse
import os
import platform
import subprocess
import sys
import textwrap

# ---------------- Utilities ----------------

def run(cmd, dryrun=False):
    print(f"> {cmd}")
    if not dryrun:
        subprocess.check_call(cmd, shell=True)

def pip_install(pkgs, dryrun=False):
    if not pkgs:
        return
    run(f'"{sys.executable}" -m pip install ' + " ".join(pkgs), dryrun)

def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ---------------- Detection ----------------

def detect_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        return name, cc
    except Exception:
        return None

def detect_python_env():
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "conda": bool(os.environ.get("CONDA_PREFIX")),
    }

# ---------------- Main Logic ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart installer for ONNX / TensorRT workflows",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--dryrun", action="store_true", help="Detect only, install nothing")
    parser.add_argument("--onnxonly", action="store_true", help="Install ONNX + CUDA stack only")
    parser.add_argument("--tensorrtonly", action="store_true", help="Install TensorRT Python bindings only")
    parser.add_argument("--full", action="store_true", help="Install ONNX + TensorRT stack")

    args = parser.parse_args()

    if not (args.dryrun or args.onnxonly or args.tensorrtonly or args.full):
        parser.error("Must specify one of --dryrun, --onnxonly, --tensorrtonly, --full")

    header("Environment Detection")

    env = detect_python_env()
    for k, v in env.items():
        print(f"{k:12}: {v}")

    gpu = detect_gpu()
    if gpu:
        name, cc = gpu
        print(f"gpu         : {name}")
        print(f"compute cap : {cc[0]}.{cc[1]}")
    else:
        print("gpu         : NOT DETECTED (CUDA unavailable)")

    # ---------------- Recommendations ----------------

    header("Recommended Stack")

    if gpu:
        major, minor = cc
        if major >= 9:
            cuda = "12.8+"
            torch = "torch nightly cu128"
            trt = "TensorRT 10.x"
        elif major >= 8:
            cuda = "12.1"
            torch = "torch stable cu121"
            trt = "TensorRT 8.6+"
        else:
            cuda = "11.8"
            torch = "torch cu118"
            trt = "TensorRT 8.x"
    else:
        cuda = "CPU"
        torch = "torch cpu"
        trt = "N/A"

    print(f"CUDA        : {cuda}")
    print(f"Torch       : {torch}")
    print(f"TensorRT    : {trt}")

    # ---------------- NVIDIA Links ----------------

    header("NVIDIA Downloads (Manual)")

    print("TensorRT:")
    print("  https://developer.nvidia.com/tensorrt")
    print()
    print("cuDNN:")
    print("  https://developer.nvidia.com/cudnn")
    print()
    print("NOTE:")
    print("  Download ZIP only. Do NOT add to system PATH.")
    print("  Runtime PATH injection will be used instead.")

    if args.dryrun:
        header("Dry Run Complete")
        print("No packages installed.")
        print("Use --onnxonly, --tensorrtonly, or --full to proceed.")
        return

    # ---------------- Install Plans ----------------

    header("Installing Dependencies")

    base_pkgs = [
        "numpy",
        "pillow",
        "opencv-python",
        "transformers",
        "tqdm",
    ]

    onnx_pkgs = [
        "onnx",
        "onnxscript",
        "onnxruntime-gpu",
    ]

    trt_pkgs = [
        "tensorrt",
        "pycuda",
    ]

    if args.onnxonly:
        print("Mode: ONNX ONLY")
        pip_install(base_pkgs + onnx_pkgs, args.dryrun)

    elif args.tensorrtonly:
        print("Mode: TENSORRT ONLY")
        pip_install(trt_pkgs, args.dryrun)

    elif args.full:
        print("Mode: FULL STACK")
        pip_install(base_pkgs + onnx_pkgs + trt_pkgs, args.dryrun)

    header("Post-Install Notes")

    print(textwrap.dedent("""
    ▶ Runtime PATH usage (recommended)

    PowerShell example:
      $env:PATH="C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\12.x\\x64;$env:PATH"
      python your_script.py

    ▶ No global PATH changes required
    ▶ Works with Conda, venv, system Python
    ▶ TensorRT ZIP stays outside your repo
    """).strip())

    header("Install Complete")

if __name__ == "__main__":
    main()
