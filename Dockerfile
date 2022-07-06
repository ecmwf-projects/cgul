FROM continuumio/miniconda3

WORKDIR /src/cgul

COPY environment.yml /src/cgul/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/cgul

RUN pip install --no-deps -e .
