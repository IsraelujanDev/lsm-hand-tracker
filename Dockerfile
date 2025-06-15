FROM python:3.11-slim

# Directory of workspace
WORKDIR /app

# install UV for dependency management
RUN pip install --no-cache-dir uv

# Copy the configuration for dependency to cache the layer
COPY pyproject.toml uv.lock ./

# Sync and install dependencies + the package (editable)
RUN uv sync --locked --no-dev --no-editable

# Copy the rest of the code
COPY . .

# Environment variable for your application
ENV LSM_BASE=/app

# Expose the Uvicorn port
EXPOSE 8000

# Start with uv run, which automatically activates the environment
CMD ["uv", "run", "uvicorn", "lsm_hand_tracker.main:app", "--host", "0.0.0.0", "--port", "8000"]
