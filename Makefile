IMAGE_NAME = mi_imagen_pytorch

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm $(IMAGE_NAME)

clean:
	docker rmi $(IMAGE_NAME)
	docker system prune -f

all: build run clean

