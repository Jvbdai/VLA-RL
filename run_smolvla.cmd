@echo off
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" vla-rl
if errorlevel 1 exit /b 1
python "%~dp0scripts\run_smolvla.py" %*
exit /b %errorlevel%
