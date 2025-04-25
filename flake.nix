{
  description = "Constrained Reinforcement Learning for Visual Locomotion from Privileged Experience";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    # Old as fuck because python 3.8 + torch for IsaacGym.
    # Upgrade to IsaacLab (which curently require 3.10) ASAP.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];
      perSystem =
        {
          lib,
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

                    # Broken and/or too long tests
                    aiohttp = python-prev.aiohttp.overrideAttrs (super: {
                      doInstallCheck = false;
                    });
                    google-crc32c = python-prev.google-crc32c.overrideAttrs (super: {
                      doInstallCheck = false;
                    });
                    moto = python-prev.moto.overrideAttrs (super: {
                      doInstallCheck = false;
                    });
                    pydevd = python-prev.pydevd.overrideAttrs (super: {
                      doInstallCheck = false;
                    });
                    pyzmq = python-prev.pyzmq.overrideAttrs (super: {
                      doCheck = false;
                    });
                    sphinx-automodapi = python-prev.sphinx-automodapi.overrideAttrs (_: {
                      doInstallCheck = false;
                    });

                    # # https://github.com/NixOS/nixpkgs/pull/288357
                    # eventlet = python-prev.eventlet.overrideAttrs (super: {
                    #   patches = (super.patches or [ ]) ++ [
                    #     (final.fetchpatch {
                    #       url = "https://github.com/eventlet/eventlet/commit/413327b229c80a97e9c89c52f7714224942701b4.patch";
                    #       hash = "sha256-rbYPd5cg3ElSYWYaZJrS7bb4nMJkTMO0ScvNnXRXzE0=";
                    #     })
                    #   ];
                    # });

                    # this is erroneously marked as no working for python < 3.9
                    flask-sqlalchemy = python-prev.flask-sqlalchemy.overrideAttrs (super: {
                      diabled = false;
                    });

                    # Missing dependencies
                    jsonschema = python-prev.jsonschema.overrideAttrs (super: {
                      propagatedBuildInputs = super.propagatedBuildInputs ++ [ python-final.pkgutil-resolve-name ];
                    });
                    jupyter-client = python-prev.jupyter-client.overrideAttrs (super: {
                      propagatedBuildInputs = super.propagatedBuildInputs ++ [ python-final.importlib-metadata ];
                    });
                    libxml2 = python-prev.libxml2.overrideAttrs (super: {
                      buildInputs = super.buildInputs ++ [ final.libxcrypt ];
                    });
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
                    openai-triton = python-prev.openai-triton.overrideAttrs (super: {
                      buildInputs = super.buildInputs ++ [ final.libxcrypt ];
                    });

                    # downgrade from 0.26.2
                    # Traceback (most recent call last):
                    #   File "train.py", line 100, in launch_rlg_hydra
                    #     PPO(cfg, envs)
                    #   File "/home/nim/local/gepetto/SoloParkour/algos/PPO.py", line 156, in PPO
                    #     next_obs = envs.reset()
                    #   File "/nix/store/x1wbk8nl356xdyg6fldayhh368qxix5h-python3-3.8.18-env/lib/python3.8/site-packages/gym/core.py", line 379, in reset
                    #     obs, info = self.env.reset(**kwargs)
                    # ValueError: not enough values to unpack (expected 2, got 1)
                    gym = python-prev.gym.overrideAttrs (super: rec {
                      version = "0.23.1";
                      src = final.fetchFromGitHub {
                        owner = "openai";
                        repo = super.pname;
                        rev = version;
                        hash = "sha256-cUVF/5RIt+igCqPqF8Fa/K0wGJS2zKj/DKfg+hi9IQY=";
                      };
                    });

                    # New packages
                    isaacgym = python-final.callPackage ./pkgs/isaacgym.nix { };
                    isaacgymenvs = python-final.callPackage ./pkgs/isaacgymenvs.nix { };
                    pkgutil-resolve-name = python-final.callPackage ./pkgs/pkgutil-resolve-name.nix { };
                    pysdf = python-final.callPackage ./pkgs/pysdf.nix { };
                    rl-games = python-final.callPackage ./pkgs/rl-games.nix { };
                    pyrender = python-final.callPackage ./pkgs/pyrender.nix { };
                    urdfpy = python-final.callPackage ./pkgs/urdfpy.nix { };
                    warp-lang = python-final.callPackage ./pkgs/warp-lang.nix { };
                  };
                };
                python38Packages = final.python38.pkgs;
              })
            ];
          };
          packages = {
            inherit (pkgs.python38Packages)
              isaacgym
              isaacgymenvs
              torch
              watchdog
              rl-games
              numpy
              gym
              ;
            default = pkgs.mkShell {
              packages = [
                pkgs.libxcrypt
                self'.packages.python
              ];
              LD_LIBRARY_PATH = "/run/opengl-driver/lib:${lib.getLib pkgs.cudaPackages.cuda_nvrtc}/lib";
            };
            python = pkgs.python38.withPackages (p: [
              p.isaacgym
              p.isaacgymenvs
              p.ipython
              # p.pybind11
              p.texttable
              p.tkinter
              p.tqdm
            ]);
          };
        };
    };
}
