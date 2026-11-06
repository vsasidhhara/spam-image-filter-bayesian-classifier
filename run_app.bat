@echo off
title DS307 Project 12 - Spam Image Filter
cd /d "%~dp0"
echo =================================================================
echo   DS307 Computer Vision: Assignment 2 (Project 12)
echo   Bayesian Spam Image Filter from Class Histograms
echo =================================================================
echo.
echo Starting Flask Web Application on http://127.0.0.1:5000 ...
echo Press Ctrl+C in this terminal to stop the server.
echo.
start http://127.0.0.1:5000
python app.py
pause
