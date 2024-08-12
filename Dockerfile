FROM mcr.microsoft.com/windows:ltsc2019

# Instala Python
WORKDIR /app

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY . /app

ADD https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe C:/python3.exe
RUN powershell -Command \
  $ErrorActionPreference = 'Stop'; \
  Start-Process c:\python3.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -Wait ; \
  Remove-Item c:\python3.exe -Force
  
# RUN python -m pip install --upgrade pip
# RUN pip install -r requirements.txt

# CMD ["python3", "main.py"]
