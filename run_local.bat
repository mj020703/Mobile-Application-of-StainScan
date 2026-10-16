@echo off
title StainScan Server
echo ============================================================
echo           Starting StainScan Server and Web Interfaces
echo ============================================================
echo.
echo Opening Mobile App: http://localhost:5000/
echo Opening Admin Dashboard: http://localhost:5000/admin.html
echo.
timeout /t 2 >nul
start "" http://localhost:5000/
start "" http://localhost:5000/admin.html
python server.py
pause
