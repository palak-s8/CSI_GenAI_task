#!/bin/bash

echo "========================================"
echo "   RAG Chat System Launcher"
echo "========================================"
echo

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "Python found! Checking dependencies..."
echo

echo "Installing/updating dependencies..."
pip3 install -r requirements.txt

echo
echo "Starting RAG Chat System..."
echo
echo "The application will open in your default browser."
echo "If it doesn't open automatically, go to: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

streamlit run app.py 