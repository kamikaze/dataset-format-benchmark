FROM python:3.13-slim-bookworm AS build-image

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN if [ -z "$ARCH" ]; then ARCH="$(uname -m)"; fi && \
    apt update && \
    apt upgrade -y && \
#    apt install -y curl ca-certificates gnupg2 && \
#    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg && \
#    echo "deb https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
#    apt update && \
    apt install -y --no-install-recommends gcc g++ make libffi-dev git cargo pkg-config
#    apt install -y --no-install-recommends gcc g++ make postgresql-server-dev-17 libpq-dev libpq5 libffi-dev git cargo pkg-config && \
#    mkdir -p /usr/lib/linux-gnu && \
#    cp /usr/lib/${ARCH}-linux-gnu/libpq.so.* \
#    /usr/lib/${ARCH}-linux-gnu/liblber-2.5.so.* \
#    /usr/lib/${ARCH}-linux-gnu/libldap-2.5.so.* \
#    /usr/lib/${ARCH}-linux-gnu/libsasl2.so.* \
#    /usr/lib/${ARCH}-linux-gnu/libgobject-2.0.so.* \
#    /usr/lib/linux-gnu/ && \
#    mkdir -p /lib/linux-gnu && \
#    cp /lib/${ARCH}-linux-gnu/libtirpc.so.* \
#    /lib/${ARCH}-linux-gnu/libnfsidmap.so.* \
#    /lib/${ARCH}-linux-gnu/libgssapi_krb5.so.* \
#    /lib/${ARCH}-linux-gnu/libkrb5.so.* \
#    /lib/${ARCH}-linux-gnu/libk5crypto.so.* \
#    /lib/${ARCH}-linux-gnu/libcom_err.so.* \
#    /lib/${ARCH}-linux-gnu/libkrb5support.so.* \
#    /lib/linux-gnu/

COPY requirements_dev.txt requirements.txt ./
RUN python3 -m pip install -U -r requirements_dev.txt && \
    python3 -m pip wheel --wheel-dir /build/wheels -r requirements.txt



FROM python:3.13-slim-bookworm AS app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY --from=build-image /build/wheels /wheels
#COPY --from=build-image /lib/linux-gnu/* /lib/linux-gnu/
#COPY --from=build-image /usr/lib/linux-gnu/* /usr/lib/linux-gnu/

#RUN if [ -z "$ARCH" ]; then ARCH="$(uname -m)"; fi && \
#    cp /lib/linux-gnu/* /lib/${ARCH}-linux-gnu/ && \
#    cp /usr/lib/linux-gnu/* /usr/lib/${ARCH}-linux-gnu/ && \
#    rm -rf /lib/linux-gnu /usr/lib/linux-gnu

WORKDIR /app

RUN python3 -m pip install -U --pre torch\<2.7.0 torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu126 && \
    python3 -m pip install --no-cache /wheels/* && \
    apt clean && \
    groupadd -r appgroup && \
    useradd -r -G appgroup -d /app appuser && \
    install -d -o appuser -g appgroup /app/logs && \
    chown -Rc appuser:appgroup /app

USER  appuser
COPY --chown=appuser . .
EXPOSE 8000

CMD ["python3", "-m", "dataset_format_benchmark"]
