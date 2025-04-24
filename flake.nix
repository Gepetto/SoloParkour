{
  description = "Constrained Reinforcement Learning for Visual Locomotion from Privileged Experience";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    # Old as fuck because python 3.8 + torch for IsaacGym.
    # Upgrade to IsaacLab (which curently require 3.10) ASAP.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      perSystem =
        {
          pkgs,
          self',
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = true;
            };
            overlays = [
              (final: prev: {
                python38 = prev.python38.override {
                  packageOverrides = python-final: python-prev: {

                    # 3.8.3 test fail
                    aiohttp = python-prev.aiohttp.overrideAttrs (super: rec {
                      version = "3.8.4";
                      src = python-final.fetchPypi {
                        inherit version;
                        inherit (super) pname;
                        hash = "sha256-vy4akWLB5EG/gFof0WbiSdV0ygTgOzT5fikodp6Rq1w=";
                      };
                    });

                    # https://github.com/NixOS/nixpkgs/pull/288357
                    eventlet = python-prev.eventlet.overrideAttrs (super: {
                      patches = (super.patches or [ ]) ++ [
                        (final.fetchpatch {
                          url = "https://github.com/eventlet/eventlet/commit/413327b229c80a97e9c89c52f7714224942701b4.patch";
                          hash = "sha256-rbYPd5cg3ElSYWYaZJrS7bb4nMJkTMO0ScvNnXRXzE0=";
                        })
                      ];
                    });

                    # https://github.com/NixOS/nixpkgs/pull/237729
                    jsonschema = python-prev.jsonschema.overrideAttrs (super: {
                      propagatedBuildInputs = super.propagatedBuildInputs ++ [ python-final.pkgutil-resolve-name ];
                    });

                    sphinx-automodapi = python-prev.sphinx-automodapi.overrideAttrs (_: {
                      doInstallCheck = false;
                    });

                    # include/python3.8/Python.h:44:10: fatal error: crypt.h: No such file or directory
                    opencv4 = python-prev.opencv4.overrideAttrs (super: {
                      buildInputs = super.buildInputs ++ [ final.libxcrypt ];
                    });
                    pyarrow = python-prev.pyarrow.overrideAttrs (super: {
                      buildInputs = super.buildInputs ++ [ final.libxcrypt ];
                    });
                    pydantic = python-prev.pydantic.overrideAttrs (super: {
                      buildInputs = super.buildInputs ++ [ final.libxcrypt ];
                    });
                    torch = python-prev.torch.overrideAttrs (super: {
                      buildInputs = super.buildInputs ++ [ final.libxcrypt ];
                    });

                    # New packages
                    easyprocess = python-final.callPackage ./pkgs/easyprocess.nix { };
                    isaacgym = python-final.callPackage ./pkgs/isaacgym.nix { };
                    isaacgymenvs = python-final.callPackage ./pkgs/isaacgymenvs.nix { };
                    pkgutil-resolve-name = python-final.callPackage ./pkgs/pkgutil-resolve-name.nix { };
                    pysdf = python-final.callPackage ./pkgs/pysdf.nix { };
                    rl-games = python-final.callPackage ./pkgs/rl-games.nix { };
                    pyrender = python-final.callPackage ./pkgs/pyrender.nix { };
                    pyvirtualdisplay = python-final.callPackage ./pkgs/pyvirtualdisplay.nix { };
                    urdfpy = python-final.callPackage ./pkgs/urdfpy.nix { };
                    warp-lang = python-final.callPackage ./pkgs/warp-lang.nix { };
                  };
                };
                python38Packages = final.python38.pkgs;
              })
            ];
          };
          packages = {
            inherit (pkgs.python38Packages) isaacgym isaacgymenvs torch;
            default = pkgs.mkShell {
              packages = [
                pkgs.libxcrypt
                self'.packages.python
              ];
              LD_LIBRARY_PATH = "/run/opengl-driver/lib";
            };
            python = pkgs.python38.withPackages (p: [
              p.isaacgym
              p.isaacgymenvs
              p.ipython
              p.pybind11
              p.texttable
              p.tkinter
              p.tqdm
            ]);
          };
        };
    };
}
