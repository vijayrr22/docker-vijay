FROM ubuntu:20.04
#Install dependencies
Run apt update && apt-get install python3 -y
CMD ["python3", "-m", "http.server", "5000"]