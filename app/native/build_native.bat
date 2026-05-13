@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%

"%~dp0..\.\.venv\Scripts\python.exe" -m cmake -S "%~dp0." -B "%~dp0build-ninja" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -Dpybind11_DIR="%~dp0..\.\.venv\Lib\site-packages\pybind11\share\cmake\pybind11" ^
  -DPython3_EXECUTABLE="%~dp0..\.\.venv\Scripts\python.exe"
if errorlevel 1 exit /b %errorlevel%

"%~dp0..\.\.venv\Scripts\python.exe" -m cmake --build "%~dp0build-ninja" --config Release
exit /b %errorlevel%
