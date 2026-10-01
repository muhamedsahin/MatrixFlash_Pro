#!/usr/bin/env python3
"""Configure/build a separate Release tree; bootstrap MSVC on Windows."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--configure-only', action='store_true')
parser.add_argument('--reconfigure', action='store_true')
parser.add_argument('--target', nargs='*', default=[])
parser.add_argument('--jobs', type=int, default=6)
args = parser.parse_args()
env = dict(os.environ)
cmake = shutil.which('cmake')
build = root / 'build' / 'perf-release'
configure = [cmake, '-S', str(root), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_CUDA_ARCHITECTURES=native', '-DMATRIX_PRO_ENABLE_CUDNN=OFF']
if os.name == 'nt':
    vsroot = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Microsoft Visual Studio'
    setups = list(vsroot.glob('*/**/VC/Auxiliary/Build/vcvars64.bat'))
    if not setups:
        raise SystemExit('MSVC vcvars64.bat not found; install Visual Studio C++ tools.')
    setup = setups[-1]
    build.mkdir(parents=True, exist_ok=True)
    bootstrap = build / 'compiler_env.cmd'
    bootstrap.write_text(f'@echo off\ncall "{setup}" >nul\nif errorlevel 1 exit /b 1\nset\n')
    output = subprocess.check_output(['cmd', '/d', '/c', str(bootstrap)],
                                     text=True, errors='replace')
    # Canonical names prevent PATH/Path duplicates in Windows child processes.
    env = {k.upper(): v for k, v in env.items()}
    env.update({line.split('=', 1)[0].upper(): line.split('=', 1)[1]
                for line in output.splitlines() if '=' in line and not line.startswith('=')})
    install = setup.parents[3]
    ninja = install / 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
    compiler = shutil.which('cl', path=env['PATH'])
    if not compiler:
        raise SystemExit('MSVC environment did not provide cl.exe')
    configure += ['-G', 'Ninja', f'-DCMAKE_MAKE_PROGRAM={ninja.as_posix()}',
                  f'-DCMAKE_CXX_COMPILER={Path(compiler).as_posix()}',
                  f'-DCMAKE_CUDA_COMPILER={Path(shutil.which("nvcc", path=env["PATH"])).as_posix()}']
if args.configure_only or args.reconfigure or not (build / 'CMakeCache.txt').exists():
    subprocess.run(configure, env=env, check=True)
if not args.configure_only:
    command = [cmake, '--build', str(build), '--parallel', str(args.jobs)]
    if args.target:
        command += ['--target', *args.target]
    subprocess.run(command, env=env, check=True)
