# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file (optional, if you have one)
COPY requirements.txt .

# Install Python libraries
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Ensure Jupyter notebook and a Python kernel are available
RUN pip install --no-cache-dir notebook ipykernel

# Copy your project files
COPY . .

# Expose Jupyter notebook port
EXPOSE 8888

# Start Jupyter Notebook with the project `notebooks` folder as the server root
#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--notebook-dir=/app/notebooks"]

ENV JUPYTER_TOKEN=personal-loan-token
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/app/notebooks --NotebookApp.token=${JUPYTER_TOKEN}"]

