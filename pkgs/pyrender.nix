{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,

  freetype-py,
  imageio,
  networkx,
  numpy,
  pillow,
  pyglet,
  pyopengl,
  pyopengl-accelerate,
  scipy,
  six,
  sphinx,
  sphinx-automodapi,
  sphinx-rtd-theme,
  trimesh,
}:

buildPythonPackage rec {
  pname = "pyrender";
  version = "0.1.45";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "mmatl";
    repo = "pyrender";
    rev = version;
    hash = "sha256-V2G8QWXMxFDQpT4XDOJhIFI2V9VhDQCaXYBb/QVLxgM=";
  };

  postPatch = ''
    substituteInPlace setup.py --replace "PyOpenGL==3.1.0" "PyOpenGL"
  '';

  build-system = [
    setuptools
  ];

  propagatedBuildInputs = [
    freetype-py
    imageio
    networkx
    numpy
    pillow
    pyglet
    pyopengl
    pyopengl-accelerate
    scipy
    six
    sphinx
    sphinx-automodapi
    sphinx-rtd-theme
    trimesh
  ];

  pythonImportsCheck = [
    "pyrender"
  ];

  doCheck = false;

  meta = {
    description = "Easy-to-use glTF 2.0-compliant OpenGL renderer for visualization of 3D scenes";
    homepage = "https://github.com/mmatl/pyrender";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
