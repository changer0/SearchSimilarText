# backend/app.py

import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
import logging
from datetime import datetime

from similarity_searcher import SimilaritySearcher

app = FastAPI(title="PDF-Excel 文本相似度检索 API")

# 配置CORS，允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为具体的前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义临时文件存放目录
TEMP_DIR = os.path.join(os.getcwd(), "scripts")
os.makedirs(TEMP_DIR, exist_ok=True)

# 定义下载文件存放目录
DOWNLOAD_DIR = os.path.join(os.getcwd(), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 配置静态文件服务，用于提供下载文件
app.mount("/downloads", StaticFiles(directory=DOWNLOAD_DIR), name="downloads")

@app.post("/process/")
async def process_files(
    source_pdf: UploadFile = File(..., description="上传的PDF文件"),
    query_excel: UploadFile = File(..., description="上传的Excel文件"),
    query_sheet_name: str = Form("Sheet1", description="Excel中查询条目所在的工作表名称"),
    query_column: str = Form("Query", description="Excel中查询条目所在的列名"),
    split_max_length: int = Form(500, description="文本分段的最大长度"),
    model_name: str = Form("BAAI/bge-large-zh-v1.5", description="用于生成嵌入的预训练模型名称"),
    top_k: int = Form(3, description="每个查询检索的相关段落数量"),
    threshold: float = Form(0.5, description="判断是否相关的相似度阈值"),
):
    try:
        # 生成唯一ID以避免文件名冲突
        unique_id = str(uuid.uuid4())
        session_dir = os.path.join(TEMP_DIR, unique_id)
        os.makedirs(session_dir, exist_ok=True)

        # 保存上传的PDF文件
        pdf_filename = f"source_{unique_id}.pdf"
        pdf_path = os.path.join(session_dir, pdf_filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(source_pdf.file, f)
        logging.info(f"保存上传的PDF文件到 {pdf_path}")

        # 保存上传的Excel文件
        excel_filename = f"query_{unique_id}.xlsx"
        excel_path = os.path.join(session_dir, excel_filename)
        with open(excel_path, "wb") as f:
            shutil.copyfileobj(query_excel.file, f)
        logging.info(f"保存上传的Excel文件到 {excel_path}")

        # 自动生成输出文件名，使用当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"FileB_with_Relevance_{timestamp}.xlsx"
        output_excel_path = os.path.join(DOWNLOAD_DIR, output_filename)

        # 创建 SimilaritySearcher 实例
        searcher = SimilaritySearcher(
            source_pdf_path=pdf_path,
            query_excel_path=excel_path,
            output_path=output_excel_path,
            query_sheet_name=query_sheet_name,
            query_column=query_column,
            split_max_length=split_max_length,
            model_name=model_name,
            top_k=top_k,
            threshold=threshold
        )

        # 运行搜索
        searcher.run()
        logging.info("SimilaritySearcher 执行成功。")

        # 检查输出文件是否生成
        if not os.path.exists(output_excel_path):
            raise HTTPException(status_code=500, detail="输出文件未生成。")

        # 生成下载URL
        download_url = f"/downloads/{output_filename}"

        # 返回下载URL
        return JSONResponse(content={
            "message": "处理完成，文件已生成。",
            "download_url": download_url
        })

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理过程中发生错误: {e}")
    finally:
        # 清理上传的文件，但保留输出文件
        try:
            shutil.rmtree(session_dir)
            logging.info(f"已删除临时目录 {session_dir}")
        except Exception as cleanup_error:
            logging.warning(f"清理临时文件时出错: {cleanup_error}")
