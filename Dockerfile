FROM golang

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && apt-get upgrade -y
RUN apt-get install -y \
  git
#  yasm \
#  python \
#  gcc \
#  g++ \
#  cmake \
#  make \
#  curl \
#  wget \
#  apt-transport-https \
#  m4 \
#  zip \
#  unzip \
#  vim \
#  build-essential

RUN git clone https://github.com/tilenmarc/SCALE-MAMBA.git
WORKDIR /SCALE-MAMBA


