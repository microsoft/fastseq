FROM nvcr.io/nvidia/pytorch:20.03-py3

MAINTAINER FastSeq Team (fastseq@microsoft.com)

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}


##############################################################################
# Mellanox OFED
##############################################################################
ENV MLNX_OFED_VERSION=4.6-1.0.1.1
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y lsb-release udev && cd ${STAGE_DIR} && \
    wget -q -O - http://www.mellanox.com/downloads/ofed/MLNX_OFED-${MLNX_OFED_VERSION}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64.tgz | tar xzf - && \
    cd MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64 && \
    ./mlnxofedinstall --user-space-only --without-fw-update --all -q && \
    cd ${STAGE_DIR} && \
    rm -rf ${STAGE_DIR}/MLNX_OFED_LINUX-${MLNX_OFED_VERSION}-ubuntu18.04-x86_64*


##############################################################################
# nv_peer_mem
##############################################################################
RUN apt-get update && \
    apt-get install -y debhelper && \
    git clone https://github.com/Mellanox/nv_peer_memory.git ${STAGE_DIR}/nv_peer_memory && \
    cd ${STAGE_DIR}/nv_peer_memory && \
    git checkout tags/1.1-0 && \
    ./build_module.sh

RUN cd ${STAGE_DIR} && \
    tar xzf ${STAGE_DIR}/nvidia-peer-memory_1.1.orig.tar.gz && \
    cd ${STAGE_DIR}/nvidia-peer-memory-1.1 && \
    apt-get install -y dkms && \
    dpkg-buildpackage -us -uc && \
    dpkg -i ${STAGE_DIR}/nvidia-peer-memory_1.1-0_all.deb


##############################################################################
# Installation/NLP Utilities
##############################################################################
RUN pip install --upgrade pip && \
	pip install yapf>=v0.30.0 && \
	pip install absl-py>=v0.9.0 && \
	pip install filelock>=v3.0.12 && \
	pip install requests>=v2.24.0 && \
	pip install gitpython>=v3.1.7 && \
	pip install rouge_score==v0.0.4 && \
	pip install fairseq==v0.9.0 \
	pip install transformers==v3.0.2 \
	pip install pytorch-transformers==1.0.0


##############################################################################
# FastSeq
##############################################################################
RUN cd / && \
    git clone https://github.com/microsoft/fastseq.git fastseq && \
    cd fastseq && \
    pip install -e .

WORKDIR "/fastseq"
