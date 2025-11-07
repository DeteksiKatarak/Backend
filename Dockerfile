# Gunakan Python image ringan
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy semua file ke container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Hugging Face Spaces (7860)
EXPOSE 7860

# Jalankan Flask app
CMD ["python", "app.py"]
