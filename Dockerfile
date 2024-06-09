# Use the jetson-inference base image
FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy your program into the container
COPY DriverAssistProgram /DriverAssistProgram

# Set the working directory
WORKDIR /DriverAssistProgram

# Install Python dependencies (if any)
RUN pip3 install -r requirements.txt

# Set the command to run your program
CMD ["python3", "ADAS.py"]
