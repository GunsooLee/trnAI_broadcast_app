FROM n8nio/n8n:1.121.1

USER root
RUN apk add --no-cache ffmpeg

# 저장용 폴더 생성 및 권한 부여
RUN mkdir -p /home/node/.n8n && chown node:node /home/node/.n8n

USER node