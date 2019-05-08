FROM continuumio/miniconda3

RUN mkdir /solar
COPY solarnet /solar/solarnet
COPY tests /solar/tests
COPY run.py /solar/
COPY *.ini /solar/
COPY environment.ubuntu.cpu.yml /solar

RUN conda env create -f /solar/environment.ubuntu.cpu.yml
