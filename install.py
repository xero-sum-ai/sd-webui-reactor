import subprocess
import os, sys
from typing import Any
import pkg_resources
from tqdm import tqdm
import urllib.request
from packaging import version as pv

try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")


BASE_PATH = os.path.dirname(os.path.realpath(__file__))

req_file = os.path.join(BASE_PATH, "requirements.txt")

models_dir = os.path.join(models_path, "insightface")

model_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
model_name = os.path.basename(model_url)
model_path = os.path.join(models_dir, model_name)

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])

def pip_uninstall(*args):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *args])

def is_installed (
        package: str, version: str | None = None, strict: bool = True
):
    has_package = None
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (installed_version != version and strict == True) or (pv.parse(installed_version) < pv.parse(version) and strict == False):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        # A missing package is the normal "please install it" case.
        # VersionConflict is raised when the argument is a requirement
        # string such as "insightface==0.7.3" and another version is present.
        if type(e).__name__ in ("DistributionNotFound", "VersionConflict"):
            return False
        print(f"Error: {e}")
        return False
    
def requirement_name(spec: str) -> str:
    spec = spec.strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in spec:
            return spec.split(sep, 1)[0].strip()
    return spec

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading...', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(model_path):
    download(model_url, model_path)

# print("ReActor preheating...", end=' ')

last_device = None
first_run = False
available_devices = ["CPU", "CUDA"]

try:
    last_device_log = os.path.join(BASE_PATH, "last_device.txt")
    with open(last_device_log) as f:
        last_device = f.readline().strip()
    if last_device not in available_devices:
        last_device = None
except:
    last_device = "CPU"
    first_run = True
    with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
        txt.write(last_device)

with open(req_file) as file:
    install_count = 0
    ort = "onnxruntime-gpu"
    import torch
    cuda_version = None
    try:
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"CUDA {cuda_version}")
            if first_run or last_device is None:
                last_device = "CUDA"
        elif torch.backends.mps.is_available() or hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
            ort = "onnxruntime"
            # to prevent errors when ORT-GPU is installed but we want ORT instead:
            if first_run:
                pip_uninstall("onnxruntime", "onnxruntime-gpu")
            # just in case:
            if last_device == "CUDA" or last_device is None:
                last_device = "CPU"
        else:
            if last_device == "CUDA" or last_device is None:
                last_device = "CPU"
        with open(os.path.join(BASE_PATH, "last_device.txt"), "w") as txt:
            txt.write(last_device)
        if cuda_version is not None:
            cuda_major = int(str(cuda_version).split(".")[0])
            # ORT 1.17.1 has no Python 3.13 wheel and no CUDA 13 build.
            # onnxruntime-gpu >= 1.27 on PyPI is the CUDA 13 package.
            modern_ort = cuda_major >= 13 or sys.version_info >= (3, 13)
            if modern_ort:
                if not is_installed("onnxruntime-gpu", "1.27.0", False):
                    install_count += 1
                    print("[ReActor] Installing onnxruntime-gpu>=1.27.0 for CUDA 13 / Python 3.13")
                    pip_uninstall("onnxruntime", "onnxruntime-gpu")
                    pip_install("onnxruntime-gpu>=1.27.0")
            else:
                if cuda_major >= 12: # CU12.x
                    extra_index_url = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
                else: # CU11.8
                    extra_index_url = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple"
                if not is_installed(ort,"1.17.1",True):
                    install_count += 1
                    ort = "onnxruntime-gpu==1.17.1"
                    pip_uninstall("onnxruntime", "onnxruntime-gpu")
                    pip_install(ort,"--extra-index-url",extra_index_url)
        elif not is_installed(ort,"1.18.1",False):
            install_count += 1
            pip_install(ort, "-U")
    except Exception as e:
        print(e)
        print(f"\nERROR: Failed to install {ort} - ReActor won't start")
        raise e
    # print(f"Device: {last_device}")
    for package in file:
        package_version = None
        strict = True
        try:
            package = package.strip()
            if not package or package.startswith("#"):
                continue
            name = requirement_name(package)
            if "==" in package:
                package_version = package.split('==', 1)[1].strip()
                strict = True
            elif ">=" in package:
                package_version = package.split('>=', 1)[1].strip()
                strict = False
            # insightface 0.7.3 does not publish a Python 3.13 wheel, and the
            # source build fails on Forge Neo. InsightFace 2.x is the package
            # that actually imports there.
            if name == "insightface" and sys.version_info >= (3, 13):
                if not is_installed("insightface", "2.0.0", False):
                    install_count += 1
                    print("[ReActor] Installing insightface>=2.1 for Python 3.13")
                    pip_install("insightface>=2.1")
                continue
            if not is_installed(name, package_version, strict):
                install_count += 1
                pip_install(package)
        except Exception as e:
            print(e)
            print(f"\nERROR: Failed to install {package} - ReActor won't start")
            raise e
    if install_count > 0:
        print(f"""
        +---------------------------------+
        --- PLEASE, RESTART the Server! ---
        +---------------------------------+
        """)
