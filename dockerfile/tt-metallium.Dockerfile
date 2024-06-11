ARG IMAGE_TAG=latest

FROM ghcr.io/tenstorrent/tt-metal/ubuntu-20.04-amd64:${IMAGE_TAG}

ARG ARCH_NAME=grayskull
ARG GITHUB_BRANCH=main

ENV ARCH_NAME=${ARCH_NAME}
ENV GITHUB_BRANCH=${GITHUB_BRANCH}

RUN git clone https://github.com/tenstorrent/tt-metal.git --depth 1 -b ${GITHUB_BRANCH} --recurse-submodules
RUN cd tt-metal \
    && pip install -e . \
    && pip install -e ttnn

SHELL ["/bin/bash", "-c"]

RUN useradd -ms /bin/bash ubuntu
USER ubuntu
WORKDIR /home/ubuntu

CMD ["tail", "-f", "/dev/null"]