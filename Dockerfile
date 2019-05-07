FROM continuumio/miniconda3

COPY solarnet /
COPY tests /
COPY run.py /
COPY *.ini /
COPY environment.ubuntu.cpu.yml /

RUN conda env create -f environment.ubuntu.cpu.yml && \
    conda activate solar

CMD bash
