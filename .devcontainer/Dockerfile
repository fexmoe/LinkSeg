# Dockerfile
FROM ubuntu:22.04

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1) Install system deps: git, ffmpeg, wget, bzip2
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ffmpeg \
      wget \
      bzip2 \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2) Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh \
&& bash /tmp/miniconda.sh -b -p $CONDA_DIR \
&& rm /tmp/miniconda.sh \
&& $CONDA_DIR/bin/conda clean -afy

# Make conda available
ENV PATH="$CONDA_DIR/bin:$PATH"

# 3) Create and prepare Python 3.9 env called "dev"
RUN conda create -y -n dev python=3.9 \
 && conda clean -afy

# Use conda’s shell init so that `conda activate` works
SHELL ["conda", "run", "-n", "dev", "/bin/bash", "-lc"]

# 4) Install Python deps (madmom + requirements.txt)
#    Assumes your requirements.txt lives at the workspace root
COPY requirements.txt /workspace/requirements.txt
RUN pip install git+https://github.com/CPJKU/madmom \
 && pip install -r /workspace/requirements.txt

# 5) Set the working directory
WORKDIR /workspace/src

# (Optional) default command
CMD ["bash"]
