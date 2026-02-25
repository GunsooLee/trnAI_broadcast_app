FROM mwader/static-ffmpeg:latest AS ffmpeg

FROM n8nio/n8n:2.8.3
USER root
COPY --from=ffmpeg /ffmpeg /usr/local/bin/
COPY --from=ffmpeg /ffprobe /usr/local/bin/

# 저장용 폴더 생성 및 권한 부여
RUN mkdir -p /home/node/.n8n && chown node:node /home/node/.n8n

USER node
