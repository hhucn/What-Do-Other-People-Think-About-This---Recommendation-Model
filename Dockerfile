FROM python:3.10

ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH "${PYTHONPATH}:/code"
ENV PYTHONHASHSEED=1

WORKDIR /code

COPY Pipfile .
COPY Pipfile.lock .
RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pipenv
RUN apt-get update && apt-get install -y libgeos-dev
RUN pipenv install --deploy --system

RUN apt-get update                             \
 && apt-get install -y --no-install-recommends \
    ca-certificates curl firefox-esr           \
 && rm -fr /var/lib/apt/lists/*

COPY . .

RUN chmod +x wait-for-it.sh
