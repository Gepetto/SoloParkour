{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,

  eigen,
  pybind11,
}:

buildPythonPackage rec {
  pname = "pysdf";
  version = "0.1.9";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-0GOYOOw63O40mdNykM2iF/wfCDTjCkFn2gSuBiCH+ns=";
  };

  postPatch = ''
    substituteInPlace setup.py \
    --replace \
      "/usr/include/eigen3" \
      "${eigen}/include/eigen3" \
    --replace \
      "if not world_ver or not major_ver or not minor_ver:" \
      "if world_ver is None or major_ver is None or minor_ver is None:"
  '';

  build-system = [
    setuptools
  ];

  nativeBuildInputs = [
    pybind11
  ];

  buildInputs = [
    eigen
  ];

  pythonImportsCheck = [
    "pysdf"
  ];

  meta = {
    description = "SDF: Convert triangle mesh to continuous signed distance function + some other mesh utilities";
    homepage = "https://pypi.org/project/pysdf/";
    license = lib.licenses.bsd2;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
