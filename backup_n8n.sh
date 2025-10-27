#!/bin/bash
# n8n 데이터 백업 스크립트

BACKUP_DIR="/home/trn/trnAi/n8n_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/n8n_backup_$TIMESTAMP.tar.gz"

# 백업 디렉토리 생성
mkdir -p "$BACKUP_DIR"

echo "🔄 n8n 데이터 백업 시작..."

# n8n 컨테이너에서 데이터 백업
docker run --rm \
  --volumes-from trnAi_n8n \
  -v "$BACKUP_DIR:/backup" \
  alpine \
  tar czf "/backup/n8n_backup_$TIMESTAMP.tar.gz" -C /home/node/.n8n .

if [ $? -eq 0 ]; then
    echo "✅ 백업 완료: $BACKUP_FILE"
    
    # 7일 이상 된 백업 삭제
    find "$BACKUP_DIR" -name "n8n_backup_*.tar.gz" -mtime +7 -delete
    echo "📁 오래된 백업 정리 완료"
else
    echo "❌ 백업 실패"
    exit 1
fi
