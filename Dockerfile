FROM golang

#ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update --fix-missing && apt-get upgrade -y
#RUN apt-get install -y \
#  git
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

RUN mkdir /go/src/github.com
RUN mkdir /go/src/github.com/fentec-project/
WORKDIR /go/src/github.com/fentec-project
RUN git clone https://github.com/tilenmarc/neural-network-on-encrypted-data.git
RUN git clone https://github.com/tilenmarc/gofe.git
WORKDIR /go/src/github.com/fentec-project/gofe
RUN git checkout dlog_for_mnist_bench
RUN go get -u -t ./...

WORKDIR /go/src/github.com/fentec-project/neural-network-on-encrypted-data
RUN git checkout linear
#RUN go get -u -t ./...

