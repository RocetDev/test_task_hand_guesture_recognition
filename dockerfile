FROM python:3.12.11-alpine

WORKDIR /app

COPY requarements.txt .

RUN pip install --upgrade pip
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requarements.txt

COPY . .

CMD ['python', 'main.py']
