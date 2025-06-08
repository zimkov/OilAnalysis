FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip с настройками для нестабильного интернета
RUN pip install --upgrade pip && \
    pip config set global.timeout 100 && \
    pip config set global.retries 10

# Сначала ставим тяжёлые пакеты с резюме-загрузкой
RUN pip install --no-cache-dir \
    --retries 10 \
    --timeout 100 \
    numpy scipy matplotlib

# Затем остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --retries 10 \
    --timeout 100 \
    -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]
