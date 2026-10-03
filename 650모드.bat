@echo off
rem 650모드: 650원을 이기면 멈춰서 기다린다 (Claude에게 "다음"이라고 하거나 logs\step.go 파일을 만들면 진행)
cd /d "%~dp0"
C:\Users\masir\.venvs\stella-sora-auto\Scripts\python.exe -u -m stella_auto.runner presets\바람.json --mode650 >> logs\night1.log 2>&1
