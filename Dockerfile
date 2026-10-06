FROM python:3.11-slim

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# WORKING DIRECTORY
# ============================================================

WORKDIR /app


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# PLAYWRIGHT
# ============================================================

RUN playwright install chromium


# ============================================================
# COPY PROJECT
# ============================================================

COPY app ./app
COPY automation ./automation
COPY jobagent ./jobagent
COPY notebook ./notebook
COPY checklist.md .
COPY data ./data


# ============================================================
# STREAMLIT
# ============================================================

EXPOSE 8501


# ============================================================
# START APPLICATION
# ============================================================

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]