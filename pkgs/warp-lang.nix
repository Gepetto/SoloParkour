{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,

  numpy,
}:

buildPythonPackage rec {
  pname = "warp-lang";
  version = "0.10.1";
  format = "wheel";

  src = fetchPypi {
    pname = "warp_lang";
    inherit version format;
    python = "py3";
    dist = "py3";
    platform = "manylinux2014_x86_64";
    hash = "sha256-Vx5qtdb3kroqM1iY4FqlZuq9Wfo1jXTDwKib7OhcS4M=";
  };

  build-system = [
    setuptools
  ];

  propagatedBuildInputs = [
    numpy
  ];

  pythonImportsCheck = [
    "warp"
  ];

  meta = {
    description = "A Python framework for high performance GPU simulation and graphics";
    homepage = "https://github.com/NVIDIA/warp";
    changelog = "https://github.com/NVIDIA/warp/blob/${src.rev}/CHANGELOG.md";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
