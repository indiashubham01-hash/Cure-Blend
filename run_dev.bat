@echo off
title CureBlend - Fullstack Dev Server
echo ========================================================
echo   Starting CureBlend (FastAPI Backend + Vite Frontend)
echo ========================================================
set "PATH=C:\Program Files\nodejs;%PATH%"

echo Checking Node.js...
node -v
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Starting Backend and Frontend concurrently...
npm run dev
pause
