## 文档相似度检索器

使用模型：
BAAI/bge-large-zh-v1.5

## 依赖安装

```
pip3 install -r requirements.txt
```

## 脚本启动

根目录使用：

```agsl
sh launcher.sh
```

## 直接后端服务


```
cd src
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

后端服务测试：

http://localhost:8000/docs

## 前端页面

http://localhost:8000/frontend/index.html

## 问题排查
## 端口占用情况
报错：
```agsl
/usr/local/bin/python3.10 /Users/lemon/PythonProject/SearchSimilarText/src/main.py 
INFO:     Will watch for changes in these directories: ['/Users/lemon/PythonProject/SearchSimilarText/src']
ERROR:    [Errno 48] Address already in use
```
查看端口占用情况：
```agsl
lsof -i :8000
```
杀死对应的进程：
```agsl
kill -9 [PID]
```