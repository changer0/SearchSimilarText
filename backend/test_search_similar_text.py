# test_search_similar_text.py

import argparse
import logging
import sys
from similarity_searcher import SimilaritySearcher

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """
    测试脚本，用于调用 SimilaritySearcher 类进行文本相似度检索。
    """
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="测试 PDF-Excel 文本相似度检索工具")
    parser.add_argument('--source_pdf_path', type=str, required=True, help="输入的来源 PDF 文件路径")
    parser.add_argument('--query_excel_path', type=str, required=True, help="输入的 Excel 文件路径（包含查询条目）")
    parser.add_argument('--output_path', type=str, required=True, help="输出的 Excel 文件路径")
    parser.add_argument('--query_sheet_name', type=str, required=False, default="Sheet1", help="Excel 文件中查询条目所在的工作表名称")
    parser.add_argument('--query_column', type=str, required=False, default="Query", help="Excel 文件中查询条目所在的列名")
    parser.add_argument('--split_max_length', type=int, required=False, default=500, help="文本分段的最大长度")
    parser.add_argument('--model_name', type=str, required=False, default='BAAI/bge-large-zh-v1.5', help="用于生成嵌入的预训练模型名称")
    parser.add_argument('--top_k', type=int, required=False, default=3, help="每个查询检索的相关段落数量")
    parser.add_argument('--threshold', type=float, required=False, default=0.5, help="判断是否相关的相似度阈值")
    
    args = parser.parse_args()
    
    try:
        # 创建 SimilaritySearcher 实例
        searcher = SimilaritySearcher(
            source_pdf_path=args.source_pdf_path,
            query_excel_path=args.query_excel_path,
            output_path=args.output_path,
            query_sheet_name=args.query_sheet_name,
            query_column=args.query_column,
            split_max_length=args.split_max_length,
            model_name=args.model_name,
            top_k=args.top_k,
            threshold=args.threshold
        )
        
        # 运行搜索
        searcher.run()
        
        logging.info("测试脚本执行完成。")
        
    except Exception as e:
        logging.error(f"在测试过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
