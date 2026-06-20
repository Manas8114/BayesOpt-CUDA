@echo off
REM Install VC++ workload to existing VS Community 2022 (elevated)
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify --quiet --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows10SDK --installPath "C:\Program Files\Microsoft Visual Studio\2022\Community"
exit /b %ERRORLEVEL%
