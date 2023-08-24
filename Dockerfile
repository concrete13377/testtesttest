FROM python:3.10.12

RUN pip install diffusers["torch"] transformers
