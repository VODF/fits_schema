# See https://testdriven.io/blog/docker-best-practices/ for recommendations
# on writing dockerfiles for python projects.

# We use a build stage to build a wheel which we then copy and install
# in the second stage to minimize image size. This is mostly needed
# because setuptools_scm needs the full version info from git and git
# itself but including that in the final image would bloat its size.

# we are using the official python image with just a version tag here
# it comes with many tools needed to build and compile python packages
# which increases it's size but is helpful for building
FROM python:3.11 AS builder

# install git for setuptools_scm
RUN apt update \
    && apt install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# add necessary sources, including .git for version info
COPY ./pyproject.toml MANIFEST.in /repo/
COPY ./src ./repo/src/
COPY ./.git ./repo/.git/

# build the wheel
RUN python -m pip install --no-cache-dir build \
    && cd repo \
    && python -m build --wheel


# second stage, copy and install wheel
# We are using the official python 3.11 image
# as base image in the slim variant to reduce image size.
FROM python:3.11-slim
COPY --from=builder /repo/dist /tmp/dist

RUN python -m pip install --no-cache-dir /tmp/dist/* \
    && rm -r /tmp/dist

RUN addgroup --system ctao && adduser --system --group ctao
USER ctao
