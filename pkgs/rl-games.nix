{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pytestCheckHook,

  poetry-core,

  gym,
  opencv4,
  psutil,
  pyyaml,
  setproctitle,
  tensorboard,
  tensorboardx,
  torch,
  wandb,
# auto-rom,
# ale-py,
# brax,
# jax,
# envpool,
# mujoco-py,
}:

buildPythonPackage rec {
  pname = "rl-games";
  version = "1.6.1";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "Denys88";
    repo = "rl_games";
    rev = "v${version}";
    hash = "sha256-qWDdMOssLPPCUBX85m7hjMPYLhSxk8C+0G2Ztr2jeAw=";
  };

  postPatch = ''
    substituteInPlace setup.py --replace "watchdog>=2.1.9,<3.0.0" "watchdog"
  '';

  build-system = [
    poetry-core
  ];

  nativeBuildInputs = [
    pytestCheckHook
  ];

  propagatedBuildInputs = [
    gym
    opencv4
    psutil
    pyyaml
    setproctitle
    tensorboard
    tensorboardx
    torch
    wandb
  ];

  # optional-dependencies = {
  #   atari = [
  #     AutoROM
  #     ale-py
  #   ];
  #   brax = [
  #     brax
  #     jax
  #   ];
  #   envpool = [
  #     envpool
  #   ];
  #   mujoco = [
  #     mujoco-py
  #   ];
  # };

  pythonImportsCheck = [
    "rl_games"
  ];

  # ??
  # > Found duplicated packages in closure for dependency 'rl-games':
  # >   rl-games 1.6.1 (/nix/store/3zcfqclaa60jq8pbkbfr3b31q1gxv8mm-python3.8-rl-games-1.6.1/lib/python3.8/site-packages)
  # >   rl-games 1.6.1 (/nix/store/3zcfqclaa60jq8pbkbfr3b31q1gxv8mm-python3.8-rl-games-1.6.1/lib/python3.8/site-packages)
  catchConflicts = false;

  meta = {
    description = "RL implementations";
    homepage = "https://github.com/Denys88/rl_games";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
