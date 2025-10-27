#!/bin/bash
# n8n 데이터 복구 스크립트

BACKUP_DIR="/home/trn/trnAi/n8n_backups"

if [ -z "$1" ]; then
    echo "사용법: ./restore_n8n.sh <백업파일명>"
    echo ""
    echo "사용 가능한 백업 파일:"
    ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "백업 파일이 없습니다."
    exit 1
fi

BACKUP_FILE="$BACKUP_DIR/$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ 백업 파일을 찾을 수 없습니다: $BACKUP_FILE"
    exit 1
fi

echo "🔄 n8n 데이터 복구 시작..."
echo "백업 파일: $BACKUP_FILE"

# n8n 컨테이너 중지
docker stop trnAi_n8n

# 데이터 복구
docker run --rm \
  --volumes-from trnAi_n8n \
  -v "$BACKUP_DIR:/backup" \
  alpine \
  sh -c "cd /home/node/.n8n && tar xzf /backup/$1"

if [ $? -eq 0 ]; then
    echo "✅ 복구 완료"
    
    # n8n 컨테이너 재시작
    docker start trnAi_n8n
    echo "🚀 n8n 재시작 완료"
    echo "📍 http://localhost:5678 접속 가능"
else
    echo "❌ 복구 실패"
    docker start trnAi_n8n
    exit 1
fi
