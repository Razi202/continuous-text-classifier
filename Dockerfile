FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8080
ENV NAME World
# Run app.py when the container launches
# CMD ["python", "app.py"]
