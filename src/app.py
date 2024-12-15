# backend/app.py

import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid
import logging
from datetime import datetime
from typing import Optional  # 导入 Optional

from similarity_searcher import SimilaritySearcher

app = FastAPI(title="PDF-Excel/Text 相似度检索 API")

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

# 配置静态文件服务，用于提供前端页面
frontend_dir = os.path.join(os.getcwd(), "frontend")
logging.info(f"frontend_dir: {frontend_dir}")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


@app.post("/process/")
async def process_files(
        source_pdf: UploadFile = File(..., description="上传的 PDF 文件"),
        query_excel: Optional[UploadFile | str] = File(None, description="上传的 Excel 文件，包含查询条目（可选）"),
        query_text: Optional[str] = Form("", description="直接上传的查询文本，每行一个查询（可选）"),
        query_sheet_name: str = Form("Sheet1", description="Excel 中查询条目所在的工作表名称"),
        query_column: str = Form("Query", description="Excel 中查询条目所在的列名"),
        split_max_length: int = Form(500, description="文本分段的最大长度"),
        embedding_model: str = Form("BAAI/bge-large-zh-v1.5", description="用于生成嵌入的预训练模型名称"),
        top_k: int = Form(3, description="每个查询检索的相关段落数量"),
        threshold: float = Form(0.5, description="判断是否相关的相似度阈值"),
):
    unique_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, unique_id)
    os.makedirs(session_dir, exist_ok=True)

    try:
        # 验证输入参数：必须提供 query_excel 或 query_text，但不能同时提供
        if (query_excel and query_text) or (not query_excel and not query_text):
            raise HTTPException(status_code=400, detail="必须上传查询的 Excel 文件或提供查询文本，且二者不能同时提供。")

        queries = []
        output_excel_path = os.path.join(DOWNLOAD_DIR,
                                         f"Output_Relevance_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")

        if query_excel:
            # 保存上传的 Excel 文件
            excel_filename = f"query_{unique_id}.xlsx"
            excel_path = os.path.join(session_dir, excel_filename)
            with open(excel_path, "wb") as f:
                shutil.copyfileobj(query_excel.file, f)
            logging.info(f"保存上传的 Excel 文件到 {excel_path}")
        else:
            # 处理上传的查询文本，按行分割
            queries = [line.strip() for line in query_text.splitlines() if line.strip()]
            if not queries:
                raise HTTPException(status_code=400, detail="提供的查询文本为空。")
            logging.info(f"使用直接上传的 {len(queries)} 个查询条目。")

        # 保存上传的 PDF 文件
        pdf_filename = f"source_{unique_id}.pdf"
        pdf_path = os.path.join(session_dir, pdf_filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(source_pdf.file, f)
        logging.info(f"保存上传的 PDF 文件到 {pdf_path}")

        # 创建 SimilaritySearcher 实例
        searcher = SimilaritySearcher(
            source_pdf_path=pdf_path,
            query_excel_path=os.path.join(session_dir, excel_filename) if query_excel else None,
            queries=queries if not query_excel else None,
            output_path=output_excel_path,
            query_sheet_name=query_sheet_name,
            query_column=query_column,
            split_max_length=split_max_length,
            model_name=embedding_model,
            top_k=top_k,
            threshold=threshold
        )

        # 运行搜索并获取 JSON 数据
        json_data = searcher.run()
        logging.info("SimilaritySearcher 执行成功。")

        # 生成下载 URL
        download_url = f"/downloads/{os.path.basename(output_excel_path)}"

        # 准备结构化的 JSON 响应
        response_data = {
            "download_url": download_url,
            "result_json": json_data
        }

        # 返回统一的 JSON 响应
        return JSONResponse(content={
            "code": 200,
            "msg": "处理完成，文件已生成。",
            "data": response_data
        })

    except HTTPException as http_exc:
        return JSONResponse(content={
            "code": http_exc.status_code,
            "msg": http_exc.detail,
            "data": None
        }, status_code=http_exc.status_code)
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        return JSONResponse(content={
            "code": 500,
            "msg": f"处理过程中发生错误: {e}",
            "data": None
        }, status_code=500)
    finally:
        # 清理上传的文件，但保留输出文件
        try:
            shutil.rmtree(session_dir)
            logging.info(f"已删除临时目录 {session_dir}")
        except Exception as cleanup_error:
            logging.warning(f"清理临时文件时出错: {cleanup_error}")
