# Miso-Helpful 🍜

A RAG-enabled personal cooking assistant built as a culmination of our Applied Machine Learning coursework.

## Overview

Miso-Helpful is an intelligent cooking assistant that leverages Retrieval-Augmented Generation (RAG) to help you with recipes, cooking techniques, and culinary questions. The chatbot provides personalized cooking guidance powered by machine learning.

## Prerequisites

- Python 3.8 or higher
- Git
- GitHub Personal Access Token (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://<YOUR_GITHUB_TOKEN>@github.com/gracey0630/miso-helpful.git
cd miso-helpful
```

Replace `<YOUR_GITHUB_TOKEN>` with your GitHub Personal Access Token.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Setup

```bash
python setup.py
```

## Running the Application

### Local Development

To run the Streamlit app locally:

```bash
streamlit run App.py
```

The application will be available at `http://localhost:8501`

### Public Access via Cloudflare Tunnel

For public access to your chatbot, use Cloudflare Tunnel:

1. **Install Cloudflare Tunnel** (Linux):
   ```bash
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Start the app with tunnel**:
   ```bash
   streamlit run App.py & cloudflared tunnel --url http://localhost:8501
   ```

3. **Access your chatbot**: Look for the Cloudflare tunnel URL in the terminal output (ends with `.trycloudflare.com`) and share that link to access the chatbot from anywhere.

## Quick Start (One Command)

```bash
pip install -r requirements.txt && python setup.py && streamlit run App.py & cloudflared tunnel --url http://localhost:8501
```

## Stopping the Application

To stop the Streamlit server:

```bash
kill $(lsof -t -i:8501)
```

## Features

- RAG-powered recipe recommendations
- Interactive cooking guidance
- Natural language processing for cooking queries
- Personalized cooking assistance

## Project Structure

```
miso-helpful/
├── App.py              # Main Streamlit application
├── setup.py            # Setup and configuration script
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

This project was developed as part of our Applied Machine Learning coursework.

