FROM python:3.9

WORKDIR /code

# Copy the requirements file to the working directory
COPY requirements-min.txt .

RUN pip install --no-cache-dir --upgrade -r /code/requirements-min.txt

COPY ./ /code/