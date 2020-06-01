FROM python:3.7.6-slim@sha256:c30d2d4ce8156922c8525d010634dfb78ad2e6de32c44dd577fa7d33305cad7e AS requirements_and_package_caching_image

WORKDIR /spotifyapp

COPY . /spotifyapp

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["run.py"]