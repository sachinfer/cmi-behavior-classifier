@echo off
echo 📊 Behavior Analytics Dashboard
echo ================================
echo.
echo Starting Streamlit application...
echo.
echo The application will be available at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run app_streamlit.py --server.port 8501
pause
