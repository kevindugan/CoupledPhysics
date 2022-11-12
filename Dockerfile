FROM ubuntu:jammy
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -yq build-essential cmake gdb libopenmpi-dev git xterm \
        texlive-latex-base texlive-latex-extra wget gmsh git-lfs && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -qO /opt/miniconda.sh && \
    # Install Miniconda
    /bin/bash /opt/miniconda.sh -b -p /opt/miniconda && \
    rm /opt/miniconda.sh && \
    # Create mpi-user
    useradd --home-dir /home/mpi-user --create-home --shell /bin/bash mpi-user && \
    mkdir /app && \
    chown -R mpi-user /app

# Create user
USER mpi-user
WORKDIR ${HOME}

# Create conda environment
SHELL ["/bin/bash", "--login", "-c"]
RUN /opt/miniconda/bin/conda init bash && \
    /opt/miniconda/bin/conda create -yqn mpi4py python=3.8 && \
    echo "conda activate mpi4py" >> ${HOME}/.bashrc && \
    /opt/miniconda/bin/conda run -n mpi4py pip install mpi4py numpy matplotlib \
        pytest-cov meshio && \
    git lfs install && \
    git config --global --add safe.directory /app

WORKDIR /app