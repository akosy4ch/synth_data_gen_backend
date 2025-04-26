# SynthData Generator Backend

This project is the backend API for generating synthetic datasets based on user-uploaded data. It supports text and numeric data generation using a variety of models, including deep learning models.

## Features
- File upload and column analysis
- Synthetic text generation (GPT-J, FLAN-T5, DeepSeek, etc.)
- Synthetic numeric generation (CTGAN, TVAE, GMM)
- Per-column model selection
- FastAPI backend with automatic docs at `/docs`
- JWT-based user authentication (register/login)
- Prepared for GPU-heavy models and scaling

## Requirements
- Python 3.10+
- pip
- (Optional) GPU for better performance with deep models

## Installation

```bash
git clone https://github.com/akosy4ch/synth_data_gen_backend.git
cd synthetic_data_generator/backend
bash setup.sh
