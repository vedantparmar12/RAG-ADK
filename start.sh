#!/bin/bash

echo "Starting Enhanced RAG Agent Backend with Advanced Features..."
echo "Features included: Query Rewriting, Semantic Chunking, ColBERT, Citations, Multi-hop, Vector Quantization, Hierarchical Search, RAGAS Evaluation"
python main.py &
BACKEND_PID=$!

echo "Starting React Frontend..."
cd frontend
npm start &
FRONTEND_PID=$!

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:3000"
echo "Press Ctrl+C to stop both services"

trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait