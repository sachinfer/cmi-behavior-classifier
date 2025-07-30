# 🐳 Docker Setup for Behavior Classification Dashboard

## 🚀 Quick Start with Docker

### Prerequisites
- **Docker** installed on your system
- **Docker Compose** (usually comes with Docker Desktop)

### 🎯 Option 1: Full AI Dashboard (Recommended)

```bash
# Build and run the full dashboard with PyTorch
docker-compose up behavior-dashboard
```

**Access at**: http://localhost:8501

### 🎯 Option 2: Demo Dashboard

```bash
# Run the demo version (faster startup)
docker-compose up behavior-dashboard-demo
```

**Access at**: http://localhost:8502

### 🎯 Option 3: Build and Run Manually

```bash
# Build the Docker image
docker build -t behavior-dashboard .

# Run the container
docker run -p 8501:8501 -v $(pwd):/app/data behavior-dashboard
```

## 📁 Project Structure with Docker

```
cmi-behavior-classifier/
│
├── Dockerfile              # 🐳 Docker configuration
├── docker-compose.yml      # 🐳 Multi-service setup
├── .dockerignore          # 🐳 Files to exclude
├── DOCKER_GUIDE.md        # 🐳 This guide
│
├── app.py                 # ✅ Full LSTM dashboard
├── app_simple.py          # ✅ Demo dashboard
├── lstm_model.pth         # ✅ Your trained model
├── requirements.txt       # ✅ Dependencies
│
└── uploads/               # 📁 Folder for CSV files (auto-created)
```

## 🔧 Docker Commands

### Build the Image
```bash
docker build -t behavior-dashboard .
```

### Run Container
```bash
# Full version
docker run -p 8501:8501 -v $(pwd):/app/data behavior-dashboard

# Demo version
docker run -p 8501:8501 -v $(pwd):/app/data behavior-dashboard python -m streamlit run app_simple.py --server.port=8501 --server.address=0.0.0.0
```

### Stop Container
```bash
# If using docker-compose
docker-compose down

# If using docker run
docker stop <container_id>
```

### View Logs
```bash
# Docker compose
docker-compose logs behavior-dashboard

# Docker run
docker logs <container_id>
```

## 🎨 What's Included in Docker

### ✅ **Full Environment:**
- **Python 3.11** with all dependencies
- **PyTorch** (CPU version for compatibility)
- **Streamlit** web framework
- **All ML libraries** (pandas, numpy, scikit-learn, etc.)
- **Your LSTM model** ready to use

### ✅ **Features:**
- **Volume mounting** for easy file access
- **Port mapping** (8501:8501)
- **Health checks** for monitoring
- **Auto-restart** on failure
- **Environment variables** configured

## 🚀 Step-by-Step Docker Setup

### 1. **Build the Image**
```bash
docker build -t behavior-dashboard .
```

### 2. **Run with Docker Compose**
```bash
# Full AI version
docker-compose up behavior-dashboard

# OR Demo version
docker-compose up behavior-dashboard-demo
```

### 3. **Access Dashboard**
- Open browser: http://localhost:8501 (full version)
- Open browser: http://localhost:8502 (demo version)

### 4. **Upload Your Data**
- Use the file uploader in the dashboard
- Or place CSV files in the `uploads/` folder

## 🔍 Troubleshooting Docker

### **Container Won't Start?**
```bash
# Check logs
docker-compose logs behavior-dashboard

# Rebuild image
docker-compose build --no-cache
```

### **Port Already in Use?**
```bash
# Change port in docker-compose.yml
ports:
  - "8503:8501"  # Use port 8503 instead
```

### **File Access Issues?**
```bash
# Check volume mounting
docker exec -it behavior-classifier-dashboard ls /app/data
```

### **Memory Issues?**
```bash
# Add memory limits to docker-compose.yml
services:
  behavior-dashboard:
    deploy:
      resources:
        limits:
          memory: 2G
```

## 🎯 Docker vs Local Setup

| Feature | Docker | Local |
|---------|--------|-------|
| **Setup Time** | ⚡ 5 minutes | 🐌 15+ minutes |
| **Dependencies** | ✅ All included | ❌ Manual install |
| **PyTorch** | ✅ Pre-installed | ❌ Manual install |
| **Portability** | ✅ Works anywhere | ❌ System specific |
| **Isolation** | ✅ Clean environment | ❌ System conflicts |
| **Updates** | ✅ Easy rebuild | ❌ Manual updates |

## 🔄 Advanced Docker Usage

### **Custom Port**
```bash
docker run -p 8080:8501 behavior-dashboard
# Access at: http://localhost:8080
```

### **Custom Data Directory**
```bash
docker run -p 8501:8501 -v /path/to/your/data:/app/data behavior-dashboard
```

### **Environment Variables**
```bash
docker run -p 8501:8501 -e STREAMLIT_SERVER_PORT=8501 behavior-dashboard
```

### **Background Mode**
```bash
docker-compose up -d behavior-dashboard
```

## 🎉 Benefits of Docker

### ✅ **No Installation Hassles**
- No need to install Python, PyTorch, or any dependencies
- Works on Windows, Mac, Linux
- No version conflicts

### ✅ **Consistent Environment**
- Same setup everywhere
- No "works on my machine" issues
- Reproducible deployments

### ✅ **Easy Sharing**
- Share the entire environment
- One command to run anywhere
- Perfect for demos and presentations

### ✅ **Production Ready**
- Can be deployed to cloud services
- Scalable and maintainable
- Professional deployment option

## 🚀 Next Steps

1. **Try Docker**: `docker-compose up behavior-dashboard`
2. **Upload Data**: Use the dashboard interface
3. **Test Predictions**: See your LSTM model in action
4. **Deploy**: Consider cloud deployment options

---

**🎯 Your AI Dashboard is Docker-Ready!**

Run it anywhere with just one command! 🐳✨ 