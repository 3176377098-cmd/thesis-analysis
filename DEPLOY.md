# 云端部署指南

两种免费方案，选一种即可。

---

## 方案一：Hugging Face Spaces（推荐，最简单）

**免费额度**: 16GB 内存 / 50GB 磁盘 / 永久免费 / 无需信用卡

### 1. 注册账号
👉 [https://huggingface.co/join](https://huggingface.co/join)

### 2. 创建 Space
1. 登录后点击右上角头像 → **+ New Space**
2. 配置选项：

| 设置项 | 选择 |
|--------|------|
| Owner | 你的用户名 |
| Space name | `paper-analyzer`（或随便起名） |
| License | MIT |
| Space SDK | **Docker** |
| Docker template | **Blank** |
| Space hardware | **CPU basic (free)** |

3. 点击 **Create Space**

### 3. 上传代码
```bash
# 在项目根目录执行:
git remote add hf https://huggingface.co/spaces/你的用户名/paper-analyzer
git push hf master
```

或者直接在 Space 页面 → **Files** → 用网页上传所有文件（除了 `data/`、`.git/`、`__pycache__/`）

### 4. 等待构建
- 首次构建约 5-8 分钟（下载依赖 + 嵌入模型）
- 构建完成后，你的系统就在 **https://你的用户名-paper-analyzer.hf.space** 运行了！

### 5. 发给别人用
直接把链接发出去即可。用户打开后：
- 左侧选 **DeepSeek**
- 填入自己的 API Key（[platform.deepseek.com](https://platform.deepseek.com) 注册获取）
- 上传论文开始分析

---

## 方案二：Docker 自部署（有服务器时用）

如果你有一台云服务器（阿里云 ECS 等），用这个方案。

### 1. 在服务器上安装 Docker
```bash
curl -fsSL https://get.docker.com | sh
```

### 2. 克隆项目并启动
```bash
git clone <你的仓库地址>
cd 基于Multi-Agent架构的学术论文深度解析系统

# 启动
docker compose up -d
```

### 3. 访问
浏览器打开 `http://你的服务器IP:8501`

---

## 配置 API Key（服务端预填，可选）

如果想让用户**不用自己填 Key**（你承担 API 费用）：

1. HuggingFace Spaces: 在 Space → **Settings → Secrets** 添加:
   ```
   DEEPSEEK_API_KEY = sk-xxxxxxxxxxxx
   ```
2. Docker: 在 `docker-compose.yml` 的 environment 里取消注释 `DEEPSEEK_API_KEY`

DeepSeek 价格约 ¥1/百万 token，个人小范围使用每月几块钱。

---

## 验证部署是否成功

启动后访问 `http://localhost:8501`，你应该看到：
- ✅ 左侧边栏有 Provider 和 API Key 输入框
- ✅ 中间有 "Upload / Analysis / Evaluation" 三个 Tab
- ✅ 命令行输出版本号无报错
