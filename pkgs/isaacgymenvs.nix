{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  python,

  # keep-sorted start
  gym,
  hydra-core,
  jinja2,
  lxml,
  omegaconf,
  pycollada,
  pyrender,
  pysdf,
  pyvirtualdisplay,
  rl-games,
  scipy,
  termcolor,
  torch,
  trimesh,
  urdfpy,
  warp-lang,
  # keep-sorted end

  isaacgym,
}:

buildPythonPackage rec {
  pname = "isaacgymenvs";
  version = "1.5.1";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "isaac-sim";
    repo = "IsaacGymEnvs";
    rev = "release/${version}";
    hash = "sha256-BAmgO7zALSIOHxeKPSpx0vakkrWh72bdndTx77KpxCc=";
  };

  postPatch = ''
    substituteInPlace setup.py \
      --replace "trimesh==3.23.5" "trimesh" \
      --replace '"hydra-core>=1.2",' ""
  '';

  # --replace "gym==0.23.1" "gym" \

  # ImportError: PyTorch was imported before isaacgym modules.  Please import torch after isaacgym modules.
  #     substituteInPlace isaacgymenvs/tasks/ant.py \
  #       --replace "import torch" "" \
  #       --replace "class Ant(VecTask):" "import torch
  # class Ant(VecTask):"

  build-system = [
    setuptools
  ];

  propagatedBuildInputs = [
    # keep-sorted start
    gym
    hydra-core
    jinja2
    lxml
    omegaconf
    pycollada
    pyrender
    pysdf
    pyvirtualdisplay
    rl-games
    scipy
    termcolor
    torch
    trimesh
    urdfpy
    warp-lang
    # keep-sorted end
  ];

  checkInputs = [
    isaacgym
  ];

  pythonImportsCheck = [
    "isaacgymenvs"
  ];

  catchConflicts = false;

  # Error: libcuda.so.1: cannot open shared object file: No such file or directory
  doCheck = false;

  # wtf ?
  postInstall = ''
    cp -r isaacgymenvs/cfg $out/${python.sitePackages}/isaacgymenvs
    cp -r isaacgymenvs/pbt/launcher/slurm $out/${python.sitePackages}/isaacgymenvs/pbt/launcher
    cp -r isaacgymenvs/tasks/{dextreme,industreal,factory} $out/${python.sitePackages}/isaacgymenvs/tasks
    cp -r isaacgymenvs/tasks/amp/poselib $out/${python.sitePackages}/isaacgymenvs/tasks/amp
  '';

  meta = {
    description = "Isaac Gym Reinforcement Learning Environments";
    homepage = "https://github.com/isaac-sim/IsaacGymEnvs";
    license = lib.licenses.bsd3;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
