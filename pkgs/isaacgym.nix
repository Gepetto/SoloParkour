{
  buildPythonPackage,
  autoPatchelfHook,

  fetchzip,

  ninja,
  python,
  libGL,
  libGLU,
  xorg,

  # python
  imageio,
  numpy,
  pillow,
  pyyaml,
  scipy,
  torch,
  torchvision,
}:
buildPythonPackage rec {
  pname = "isaacgym";
  version = "1.0.preview4";

  src = fetchzip {
    url = "https://developer.nvidia.com/isaac-gym-preview-4";
    hash = "sha256-MnhVhflsqSrS/9DuYjFzvzHV2hwLLrREBAPa479ccxY=";
    extension = "tar.gz";
  };
  sourceRoot = "${src.name}/python";
  postPatch = ''
    substituteInPlace setup.py --replace '"ninja",' ""

    # drop python 3.6 / 3.7 stuff
    pushd isaacgym/_bindings/linux-x86_64
    rm -rf \
      py36 \
      gym_36.so \
      rlgpu_36.so \
      gym_37.so \
      rlgpu_37.so 
    popd
  '';

  nativeBuildInputs = [
    autoPatchelfHook
  ];
  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
    "libRobotImp.so" # no idea
  ];

  buildInputs = [
    xorg.libX11
    libGL
    libGLU
  ];

  propagatedBuildInputs = [
    imageio
    ninja
    numpy
    pillow
    pyyaml
    scipy
    torch
    torchvision
  ];

  # This is somehow looking into src dir where ELF are not patched
  doCheck = false;

  postInstall = ''
    cp -r isaacgym/_bindings/src $out/${python.sitePackages}/isaacgym/_bindings/
  '';

  meta = {
    description = "GPU-accelerated simulation and reinforcement learning toolkit";
    license = "Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.";
  };
}
