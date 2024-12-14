## 查找相似文本

使用模型：

BAAI/bge-large-zh-v1.5

BAAI/bge-large-zh-v1.5

## 运行后端服务

```
cd backend
pip3 install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

后端服务测试：

http://localhost:8000/docs


遗留问题：如果分段为 0 需要关注！
