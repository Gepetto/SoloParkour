{
  lib,
  buildPythonPackage,
  fetchFromGitHub,

  setuptools,

  freetype-py,
  imageio,
  lxml,
  networkx,
  numpy,
  pillow,
  pycollada,
  pyglet,
  pyopengl,
  pyrender,
  scipy,
  six,
  sphinx,
  sphinx-automodapi,
  sphinx-rtd-theme,
  trimesh,

  pytest,
}:

buildPythonPackage rec {
  pname = "urdfpy";
  version = "0.0.22";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "mmatl";
    repo = "urdfpy";
    rev = version;
    hash = "sha256-iF15pglvulrvq+4HdqTGLd/HnX2dBicxLZ67g449CpE=";
  };

  postPatch = ''
    substituteInPlace setup.py \
      --replace "pycollada==0.6" "pycollada" \
      --replace "networkx==2.2" "networkx"
  '';

  build-system = [
    setuptools
  ];

  nativeBuildInputs = [
    freetype-py
    imageio
    lxml
    networkx
    numpy
    pillow
    pycollada
    pyglet
    pyopengl
    pyrender
    scipy
    six
    sphinx
    sphinx-automodapi
    sphinx-rtd-theme
    trimesh
  ];

  checkInputs = [
    pytest
  ];

  pythonImportsCheck = [
    "urdfpy"
  ];

  meta = {
    description = "Python parser for URDFs";
    homepage = "https://github.com/mmatl/urdfpy";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
