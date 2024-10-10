# Use an official Python runtime as a parent image
FROM mcr.microsoft.com/cbl-mariner/base/python:3

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set environment variables
ENV AOAI_TYPE=_  # AOAI_TYPE in .ENV
ENV AOAI_BASE=_  # AOAI_BASE in .ENV
ENV AOAI_VERSION=_  # AOAI_VERSION in .ENV
ENV AOAI_KEY=_  # AOAI_KEY in .ENV
ENV AOAI_ENGINE=_ # AOAI_ENGINE in .ENV

# Make port 5000 available to the world outside this container
EXPOSE 2000

# Run app.py when the container launches
CMD ["python3", "main.py"]
