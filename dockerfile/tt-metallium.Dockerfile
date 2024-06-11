ARG IMAGE_TAG=latest
ARG ARCH_NAME=grayskull
ARG GITHUB_BRANCH=main

FROM ghcr.io/tenstorrent/tt-metal/ubuntu-20.04-amd64:${IMAGE_TAG}
USER ubuntu

ENV ARCH_NAME=${ARCH_NAME}

RUN git clone https://github.com/tenstorrent/tt-metal.git --depth 1 -b ${GITHUB_BRANCH} /opt/tt-metal --recurse-submodules
RUN cd tt-metal \
    && pip install -e . \
    && pip install -e ttnn
CMD ["tail", "-f", "/dev/null"]