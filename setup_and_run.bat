@echo off
REM Quick start script for Windows (PowerShell-friendly).
REM Usage:
REM   setup_and_run.bat install     -> install deps (CPU torch by default)
REM   setup_and_run.bat install_gpu -> install with CUDA 12.1 wheels
REM   setup_and_run.bat train       -> run training on GPU
REM   setup_and_run.bat web         -> launch Flask app at http://127.0.0.1:5000
REM   setup_and_run.bat all         -> install (CPU) + launch web

set ROOT=%~dp0
set ACTION=%1
if "%ACTION%"=="" set ACTION=all

if /I "%ACTION%"=="install" goto install_cpu
if /I "%ACTION%"=="install_gpu" goto install_gpu
if /I "%ACTION%"=="train" goto train
if /I "%ACTION%"=="web" goto web
if /I "%ACTION%"=="all" goto allflow
echo Unknown action: %ACTION%
exit /b 1

:install_cpu
python -m pip install --upgrade pip
python -m pip install -r "%ROOT%requirements_pytorch.txt" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
goto :eof

:install_gpu
python -m pip install --upgrade pip
python -m pip install -r "%ROOT%requirements_pytorch.txt" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
goto :eof

:train
python -m pytorch_impl.train --data "%ROOT%data\lol_dataset" --img-size 512 --epochs 180 --batch-size 1 --critic-updates 5 --save-dir "%ROOT%weights"
goto :eof

:web
set EDNIG_HOST=127.0.0.1
set EDNIG_PORT=5000
python "%ROOT%webapp\app.py"
goto :eof

:allflow
call :install_cpu
call :web
goto :eof
