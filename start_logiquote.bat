@echo off
title LogiQuote System

:: Kích hoạt môi trường ảo (sửa lại đường dẫn nếu khác)
call C:\Users\ADMIN\anaconda3\Scripts\activate.bat logiquote

:: Chạy API (FastAPI) trên cổng 8000
start cmd /k "uvicorn api:app --reload --port 8000"

:: Chạy UI (Streamlit) trên cổng 8501
start cmd /k "streamlit run app.py --server.port 8501"

echo.
echo =========================================
echo  🚀 Hệ thống LogiQuote đã khởi động
echo  - API chạy tại: http://127.0.0.1:8000/docs
echo  - UI  chạy tại: http://localhost:8501
echo =========================================
pause
