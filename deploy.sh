#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh  —  一键将本地代码打包上传到腾讯云服务器并重启服务
#
# 用法:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# 首次使用前，在下方填写服务器信息：
# ─────────────────────────────────────────────────────────────────────────────

# ── 配置区（只需改这里）─────────────────────────────────────────────────────
SERVER_USER="ubuntu"                    # 服务器 SSH 用户名，腾讯云默认 ubuntu 或 root
SERVER_HOST="106.55.107.115"           # 服务器公网 IP 或域名
SERVER_PORT="22"                        # SSH 端口，默认 22
REMOTE_DIR="/opt/dialogue-eval"         # 服务器上的部署目录
SSH_KEY=""                              # SSH 私钥路径，留空则用默认 (~/.ssh/id_rsa)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="/tmp/dialogue-eval-deploy.tar.gz"

# Build SSH/SCP options (ssh uses -p, scp uses -P for port)
SSH_OPTS="-p ${SERVER_PORT} -o StrictHostKeyChecking=no"
SCP_OPTS="-P ${SERVER_PORT} -o StrictHostKeyChecking=no"
if [ -n "$SSH_KEY" ]; then
  SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
  SCP_OPTS="$SCP_OPTS -i $SSH_KEY"
fi

echo "────────────────────────────────────────"
echo "  Dialogue Eval — 部署脚本"
echo "  目标: ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}"
echo "────────────────────────────────────────"

# ── Step 1: 构建前端 ────────────────────────────────────────────────────────
echo "[1/5] 构建前端..."
cd "$PROJECT_DIR/web"
npm run build
cd "$PROJECT_DIR"

# ── Step 2: 打包（排除不必要的文件）────────────────────────────────────────
echo "[2/5] 打包项目文件..."
tar -czf "$ARCHIVE" \
  --exclude='.git' \
  --exclude='.env' \
  --exclude='web/node_modules' \
  --exclude='web/src' \
  --exclude='web/public' \
  --exclude='**/__pycache__' \
  --exclude='**/*.pyc' \
  --exclude='.DS_Store' \
  --exclude='reports' \
  --exclude='.pytest_cache' \
  --exclude='tests' \
  --exclude='web/index.html' \
  --exclude='web/tsconfig*' \
  --exclude='web/vite.config.ts' \
  --exclude='web/package*' \
  -C "$PROJECT_DIR" .

echo "   打包完成: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"

# ── Step 3: 上传到服务器 ────────────────────────────────────────────────────
echo "[3/5] 上传到服务器..."
scp $SCP_OPTS "$ARCHIVE" "${SERVER_USER}@${SERVER_HOST}:/tmp/dialogue-eval-deploy.tar.gz"

# ── Step 4: 服务器端部署 ────────────────────────────────────────────────────
echo "[4/5] 服务器端部署..."
ssh $SSH_OPTS "${SERVER_USER}@${SERVER_HOST}" bash <<REMOTE_SCRIPT
set -euo pipefail

# 安装 Docker（如果没有）
if ! command -v docker &> /dev/null; then
  echo "  安装 Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker \$USER || true
fi

# 安装 Docker Compose 插件（如果没有）
if ! docker compose version &> /dev/null 2>&1; then
  echo "  安装 Docker Compose 插件..."
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-compose-plugin
fi

# 创建部署目录
sudo mkdir -p "${REMOTE_DIR}"
sudo chown \$(whoami):\$(whoami) "${REMOTE_DIR}"

# 解压（保留已有的 .env 和 reports/）
cd "${REMOTE_DIR}"
tar -xzf /tmp/dialogue-eval-deploy.tar.gz
rm /tmp/dialogue-eval-deploy.tar.gz

# 如果还没有 .env，从模板创建
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "  ⚠️  首次部署：请编辑 ${REMOTE_DIR}/.env 填写 API Key，然后重新运行:"
  echo "      cd ${REMOTE_DIR} && docker compose up -d"
  exit 0
fi

# 重启服务
echo "  重启服务..."
docker compose pull 2>/dev/null || true
docker compose up -d --build

echo "  ✅ 部署完成！服务运行在 http://\$(curl -s ifconfig.me):8000"
REMOTE_SCRIPT

# ── Step 5: 清理本地临时文件 ────────────────────────────────────────────────
echo "[5/5] 清理临时文件..."
rm -f "$ARCHIVE"

echo ""
echo "════════════════════════════════════════"
echo "  ✅ 部署完成"
echo "  访问地址: http://${SERVER_HOST}:8000"
echo "════════════════════════════════════════"
