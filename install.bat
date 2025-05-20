
@echo off
setlocal EnableDelayedExpansion

echo Installing Orion Language...

:: Define installation directory
set "Orion_DIR=C:\Orion"

:: Create directory if not exists
if not exist "%Orion_DIR%" mkdir "%Orion_DIR%"

:: Copy the interpreter to installation directory
xcopy /Y "%~dp0interpreter.py" "%Orion_DIR%"

:: Create a batch file to run the interpreter
(echo @echo off
 echo python "%Orion_DIR%\interpreter.py" %%*) > "%Orion_DIR%\orion.bat"

:: Add directory to system PATH
setx PATH "%Orion_DIR%;!PATH!" /M

:: Verify installation
echo Installation complete. You can now use 'orion run <filename.pl>' or 'orion shell' in CMD.
endlocal
