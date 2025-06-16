# SEME Project (同济大学软件工程管理与经济学项目)

> 2025年同济大学软件工程管理与经济学项目

## 项目概述 | Project Overview

这是一个基于Vue3和Flask的全栈Web应用项目，采用前后端分离架构。项目包含前端用户界面和后端API服务，使用Python进行深度学习模型预测。

This is a full-stack web application project based on Vue3 and Flask, using a front-end and back-end separated architecture. The project includes a front-end user interface and back-end API services, with Python-based deep learning model predictions.

## 项目结构 | Project Structure

```
├── frontend/                # 前端项目目录
│   ├── src/                # 源代码目录
│   ├── index.html          # 入口HTML文件
│   ├── package.json        # 前端依赖配置
│   └── vite.config.js      # Vite配置文件
│
├── backend/                # 后端项目目录
│   ├── app/               # 应用主目录
│   │   ├── auth/         # 认证模块
│   │   ├── patient/      # 病人管理模块
│   │   ├── mri/          # MRI相关模块
│   │   ├── prediction/   # 预测模块
│   │   └── models.py     # 数据模型
│   ├── tests/            # 测试目录
│   ├── environment.yml   # Conda环境配置
│   ├── app.py           # 应用入口
│   └── wsgi.py          # WSGI服务器配置
│
└── README.md             # 项目说明文档
```

## 环境要求 | Requirements

### 前端 | Frontend
- Node.js >= 16.0.0
- npm >= 8.0.0

### 后端 | Backend
- Python 3.11
- Conda (用于环境管理)
- CUDA支持 (用于深度学习模型)

## 安装和运行 | Installation & Running

### 后端设置 | Backend Setup

1. 创建并激活Conda环境 | Create and activate Conda environment
```bash
conda env create -f backend/environment.yml
conda activate se
```

2. 启动后端服务 | Start backend server
```bash
cd backend
python wsgi.py
```
后端服务将在 http://localhost:5000 运行

### 前端设置 | Frontend Setup

1. 安装依赖 | Install dependencies
```bash
cd frontend
npm install
```

2. 启动开发服务器 | Start development server
```bash
npm run dev
```
前端开发服务器将在 http://localhost:5173 运行

3. 构建生产版本 | Build for production
```bash
npm run build
```

## 测试 | Testing

运行后端测试 | Run backend tests:
```bash
cd backend
python -m pytest tests/ -v
```

## 项目特性 | Features

- 现代化的前端框架 (Vue 3 + Vite)
- RESTful API后端服务
- JWT认证
- 深度学习模型集成
- 完整的测试套件

## 注意事项 | Notes

- 确保已正确配置数据库连接
- 深度学习模型需要CUDA支持
- 上传文件存储在backend/uploads目录

## 许可证 | License

[MIT License](LICENSE)

---
© 2025 同济大学软件工程管理与经济学项目组
