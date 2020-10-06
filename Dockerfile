#I specify the parent base image which is the python version 3.7
FROM debian

FROM python:3.7

## FROM python:3.7-slim-buster

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
#RUN pip install --no-cache-dir  --upgrade pip

# set work directory
WORKDIR /src/app/ecoplantContainer

# copy requirements.txt
COPY requirements.txt /src/app/ecoplantContainer/requirements.txt

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# set work directory
WORKDIR /src/app/ecoplantContainer

# set app port
EXPOSE 8080

ENTRYPOINT [ "python" ]

# Run app.py when the container launches
CMD [ "app.py","run","--host","0.0.0.0"]