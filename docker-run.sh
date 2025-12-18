#!/bin/bash
# Docker 运行脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Seq2Seq NER+RE Docker 运行脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}错误: Docker 未安装，请先安装 Docker${NC}"
    exit 1
fi

# 检查是否有 GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}检测到 NVIDIA GPU，使用 GPU 版本${NC}"
    USE_GPU=true
else
    echo -e "${YELLOW}未检测到 NVIDIA GPU，使用 CPU 版本${NC}"
    USE_GPU=false
fi

# 创建必要的目录
mkdir -p saved_model logs

# 构建和运行
if [ "$USE_GPU" = true ]; then
    echo -e "${GREEN}构建 GPU Docker 镜像...${NC}"
    docker-compose -f docker-compose.gpu.yml build
    
    echo -e "${GREEN}启动容器（GPU 模式）...${NC}"
    docker-compose -f docker-compose.gpu.yml up
else
    echo -e "${GREEN}构建 CPU Docker 镜像...${NC}"
    docker-compose build
    
    echo -e "${GREEN}启动容器（CPU 模式）...${NC}"
    docker-compose up
fi

