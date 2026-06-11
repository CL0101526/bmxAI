# BMX Coach AI

An AI-powered computer vision system for analyzing and coaching BMX racing techniques in real-time.

## Features

- **Pose Estimation**: Uses YOLOv11 pose detection to track rider and bike geometry
- **Skill Analysis**: Supports pump, manual, step-up jump, and double-jump techniques
- **Elite Benchmarking**: Compares rider performance against synthetic elite baselines
- **AI Coaching**: Local Llama 3 integration for personalized coaching feedback
- **Fast DTW Scoring**: Dynamic Time Warping algorithm for technique matching

## Setup

### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics YOLOv11
- FastAPI & Uvicorn
- Ollama (for local LLM coaching)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Application

```bash
python main.py
```

The app will start on `http://127.0.0.1:8000`

## Supported Skills

- **Pump**: Hip-hinge angle analysis on rollers
- **Manual**: Hip-to-rear-axle offset balance tracking
- **Step-Up Jump**: Bike pitch (front/rear hub) on take-off
- **Double Jump**: Pop speed and trajectory analysis

## Architecture

- **main.py**: Core analysis pipeline and FastAPI server
- **index.html**: Web UI for video upload and analysis
- **benchmarks/**: Directory for elite reference videos
