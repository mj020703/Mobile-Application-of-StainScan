@echo off
echo ===================================================
echo   StainScan - Enabling Local Mobile Network Access
echo ===================================================
echo.
echo 1. Adding Windows Firewall Rule for Port 5000...
netsh advfirewall firewall add rule name="StainScan Port 5000" dir=in action=allow protocol=TCP localport=5000
echo.
echo 2. Setting 'Sir Arnold' Wi-Fi to Private Network...
powershell -NoProfile -Command "Set-NetConnectionProfile -Name 'Sir Arnold' -NetworkCategory Private -ErrorAction SilentlyContinue"
echo.
echo ===================================================
echo   DONE! Port 5000 is now accessible on your Wi-Fi.
echo   You can now open on your phone:
echo   http://192.168.1.56:5000
echo ===================================================
pause
