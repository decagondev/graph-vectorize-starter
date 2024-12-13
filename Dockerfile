FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION=1.7.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi

COPY . .

RUN addgroup --system --gid 1001 graphuser \
    && adduser --system --uid 1001 --ingroup graphuser graphuser

RUN chown -R graphuser:graphuser /app

USER graphuser

CMD ["poetry", "run", "python", "main.py"]
