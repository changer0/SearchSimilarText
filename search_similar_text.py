# 导入所需的库
import pdfplumber          # 用于从 PDF 中提取文本
import pandas as pd        # 用于处理 Excel 文件
import re                  # 用于正则表达式操作
from sentence_transformers import SentenceTransformer  # 用于生成句子嵌入
import faiss               # 用于高效的向量检索
import numpy as np         # 用于数值计算
import json                # 用于处理 JSON 数据
import logging             # 用于日志记录
import argparse            # 用于解析命令行参数

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 函数：从 PDF 文件中提取文本内容
def extract_text_from_pdf(source_pdf_path):
    text = ""
    try:
        with pdfplumber.open(source_pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        logging.info(f"成功提取 PDF 文件中的文本内容。")
    except Exception as e:
        logging.error(f"提取 PDF 内容时出错: {e}")
        raise e
    return text

# 函数：读取 Excel 文件中的查询条目
def read_excel_queries(query_excel_path, query_sheet_name=0, query_column='Query'):
    try:
        df = pd.read_excel(query_excel_path, sheet_name=query_sheet_name)
        queries = df[query_column].dropna().tolist()
        logging.info(f"成功读取 Excel 文件中的 {len(queries)} 个查询条目。")
    except Exception as e:
        logging.error(f"读取 Excel 文件时出错: {e}")
        raise e
    return queries, df

# 函数：将文本按指定的最大长度分段
def split_text(text, spilt_max_length=500):
    sentences = re.split(r'(?<=[。！？；])', text)
    segments = []
    current_segment = ""
    for sentence in sentences:
        if len(current_segment) + len(sentence) <= spilt_max_length:
            current_segment += sentence
        else:
            segments.append(current_segment)
            current_segment = sentence
    if current_segment:
        segments.append(current_segment)
    logging.info(f"成功将文本分段为 {len(segments)} 个段落。")
    return segments

# 函数：将嵌入向量归一化为单位向量
# embeddings：一个二维 NumPy 数组，形状为 (num_vectors, dim)，其中 num_vectors 是向量的数量，dim 是每个向量的维度。
def normalize_embeddings(embeddings):
    """
    函数的主要目的是将嵌入向量（embeddings）归一化为单位向量。归一化后的向量长度（或范数）为1。这在许多应用场景中非常重要，特别是在计算向量相似度（如余弦相似度）时。
    """
    # 计算每个向量的范数（即长度）。
    # keepdims=True：保持输出数组的维度。结果 norms 的形状为 (num_vectors, 1)，而不是 (num_vectors,)。这样在后续的除法操作中，维度会自动广播，避免形状不匹配的问题。
    # axis=1：沿着第二个轴（即每一行）计算范数。这样每一行的向量会被单独计算其范数。
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 防止除以零
    # 将每个向量除以其范数，从而得到归一化后的单位向量。
    # 由于 norms 的形状为 (num_vectors, 1)，而 embeddings 的形状为 (num_vectors, dim)，NumPy 会自动将 norms 广播到每一行，从而实现逐元素除法。
    normalized_embeddings = embeddings / norms
    logging.info("normalize_embeddings 成功归一化嵌入向量。")
    return normalized_embeddings

# 函数：生成文本段落的嵌入向量
def generate_embeddings(segments, model_name='BAAI/bge-large-zh-v1.5'):
    """
    generate_embeddings 函数的主要目的是将文本段落（segments）转换为数值向量（嵌入向量）
    ，以便在后续的相似度计算和检索任务中使用。
    """
    model = SentenceTransformer(model_name)
    # convert_to_tensor：指定输出为 NumPy 数组而非张量（Tensor）。这样做是为了兼容后续使用的 FAISS 库，FAISS 通常使用 NumPy 数组进行向量索引和搜索。
    # show_progress_bar：显示进度条，以了解程序运行状态。
    embeddings = model.encode(segments, convert_to_tensor=False, show_progress_bar=True)
    # 将嵌入向量转换为 float32 类型的 NumPy 数组。
    # FAISS 要求输入的嵌入向量为 float32 类型。如果嵌入向量的类型不同（如 float64），可能会导致兼容性问题或性能下降。
    embeddings = np.array(embeddings).astype('float32')
    # 目的是确保所有嵌入向量都是单位向量，以方便后续的相似度计算。
    embeddings = normalize_embeddings(embeddings)
    return embeddings, model

# 函数：构建 FAISS 内积索引，用于余弦相似度检索
# embeddings: 一个二维 NumPy 数组，形状为 (num_vectors, dim)，其中 num_vectors 是向量的数量，dim 是每个向量的维度。
def build_faiss_index_cosine(embeddings):
    """
    函数的主要目的是使用 FAISS（Facebook AI Similarity Search）库创建一个内积索引，
    以便后续进行基于 余弦相似度 的高效向量检索。
    具体步骤包括确定向量的维度、创建内积索引对象、将嵌入向量添加到索引中，并返回构建好的索引对象。
    """
    try:
        dimension = embeddings.shape[1]
        logging.info(f"使用 {dimension} 维度构建 FAISS 内积索引。")
        # FAISS 中的 IndexFlatIP 是一个“扁平”索引，使用内积作为相似度度量，适用于基于余弦相似度的检索任务。
        # 扁平索引（Flat Index）：意味着所有的向量都存储在内存中，没有进行任何压缩或分层索引。适用于中小规模的数据集，保证了检索的准确性，但在大规模数据集上可能会受到内存和计算资源的限制。
        # 如果使用内积索引，就必须进行向量归一化，否则影响准确性
        index = faiss.IndexFlatIP(dimension)  # 使用内积索引
        index.add(embeddings)
        logging.info(f"成功构建 FAISS 内积索引，包含 {index.ntotal} 个向量。")
    except Exception as e:
        logging.error(f"构建 FAISS 内积索引时出错: {e}")
        raise e
    return index

# 函数：检索与查询最相关的文本段落，基于余弦相似度
def retrieve_cosine_similarity(query, model, index, segments, top_k=1):
    try:
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        query_embedding = normalize_embeddings(query_embedding)
    except Exception as e:
        logging.error(f"生成查询嵌入向量时出错: {e}")
        raise e
    
    try:
        similarities, indices = index.search(query_embedding, top_k)
    except Exception as e:
        logging.error(f"使用 FAISS 进行检索时出错: {e}")
        raise e
    
    results = []
    for similarity, idx in zip(similarities[0], indices[0]):
        if idx < len(segments):
            results.append({
                'segment': segments[idx],
                'similarity': similarity  # 余弦相似度
            })
    return results

# 函数：移除 Excel 不允许的非法字符
def remove_illegal_characters(text):
    """
    移除 Excel 不允许的非法字符（Unicode 码点小于 32，除了一些允许的字符）。
    
    参数:
    - text (str): 输入的字符串。
    
    返回:
    - clean_text (str): 清洗后的字符串。
    """
    if not isinstance(text, str):
        return text
    # 定义非法字符的正则表达式，除去允许的字符（如制表符、换行符、回车符）
    illegal_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    clean_text = illegal_chars.sub('', text)
    return clean_text

# 函数：检查每个查询条目在 PDF 中的相关性，并生成输出
def check_relevance_and_output_cosine(queries, model, index, segments, df_b, top_k=1, threshold=0.5):
    """
    对每个查询条目进行相关性检索，并在 DataFrame 中标注是否有相关内容及相关段落。
    
    参数:
    - queries (list): 查询条目列表。
    - model (SentenceTransformer): 用于生成嵌入的模型。
    - index (faiss.IndexFlatIP): FAISS 内积索引。
    - segments (list): 存储所有文本段落的列表。
    - df_b (DataFrame): 包含查询条目的原始 Excel 数据。
    - top_k (int): 每个查询检索的相关段落数量。默认值为 1。
    - threshold (float): 判断是否相关的相似度阈值。默认值为 0.5。
    
    返回:
    - df_b (DataFrame): 更新后的 DataFrame，新增 'Has_Relevance_in_FileA', 'Related_Segments' 和 'Similarities' 列。
    """
    relevance = []          # 存储每个查询的相关性结果（True/False）
    related_segments = []   # 存储每个查询的相关段落内容
    related_similarities = []  # 存储每个查询的相关段落相似度

    for query in queries:
        logging.info(f"正在处理查询: {query}")
        # 检索当前查询最相关的段落
        results = retrieve_cosine_similarity(query, model, index, segments, top_k=top_k)
        # 判断是否有任何一个检索结果的相似度超过阈值，从 results 中查询 similarity 大于设定阈值的场景
        has_relevance = any(res['similarity'] >= threshold for res in results)
        relevance.append(has_relevance)
        
        # 如果有相关性，收集相关段落和相似度
        if has_relevance:
            # 筛选出相似度超过阈值的段落
            relevant_results = [res for res in results if res['similarity'] >= threshold]
            # 格式化相关段落和相似度，增加结构化
            formatted_segments = ""
            formatted_similarities = ""
            for i, res in enumerate(relevant_results, start=1):
                formatted_segments += f"第{i}条相似段落：\n\n {res['segment']}\n——————————————————————————————————————————————————————\n\n"
                formatted_similarities += f"第{i}条相似段落相似度: {res['similarity']:.4f}\n"
        else:
            formatted_segments = ""
            formatted_similarities = ""
        
        related_segments.append(formatted_segments.strip())       # 去除末尾多余的换行
        related_similarities.append(formatted_similarities.strip())

    # 将相关性结果添加到 DataFrame 中
    df_b['Has_Relevance_in_FileA'] = relevance
    # 将相关段落内容和相似度添加到 DataFrame 中
    df_b['Related_Segments'] = related_segments
    df_b['Similarities'] = related_similarities
    logging.info("完成所有查询的相关性检索。")
    return df_b

# 主函数：整合所有步骤，实现完整的流程
def main():
    """
    主函数，执行从 PDF 提取文本、读取 Excel 查询、生成嵌入、构建索引、
    检索相关性并输出结果的完整流程。
    """
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="PDF-Excel 文本相似度检索脚本")
    parser.add_argument('--source_pdf_path', type=str, required=True, help="输入的来源 PDF 文件路径")
    parser.add_argument('--query_excel_path', type=str, required=True, help="输入的 Excel 文件路径（包含查询条目）")
    parser.add_argument('--output_path', type=str, required=True, help="输出的 Excel 文件路径")
    # parser.add_argument('--faiss_index_path', type=str, required=False, default="faiss_index.index", help="FAISS 索引文件的保存路径")
    parser.add_argument('--query_sheet_name', type=str, required=False, default="Sheet1", help="Excel 文件中查询条目所在的工作表名称")
    parser.add_argument('--query_column', type=str, required=False, default="Query", help="Excel 文件中查询条目所在的列名")
    parser.add_argument('--spilt_max_length', type=int, required=False, default=500, help="文本分段的最大长度")
    parser.add_argument('--model_name', type=str, required=False, default='BAAI/bge-large-zh-v1.5', help="用于生成嵌入的预训练模型名称")
    parser.add_argument('--top_k', type=int, required=False, default=3, help="每个查询检索的相关段落数量")
    parser.add_argument('--threshold', type=float, required=False, default=0.5, help="判断是否相关的相似度阈值")
    
    args = parser.parse_args()
    
    try:
        # 步骤 1: 提取 FileA 的文本内容
        logging.info("提取 PDF 的文本内容...")
        file_a_text = extract_text_from_pdf(args.source_pdf_path)
        
        # 步骤 2: 分段 FileA 的文本内容
        logging.info("分段 PDF 的文本内容...")
        file_a_segments = split_text(file_a_text, spilt_max_length=args.spilt_max_length)  # 可根据需要调整 spiltspilt_max_length
        
        # 步骤 3: 生成 FileA 的嵌入向量，并归一化
        logging.info("生成 PDF 的嵌入向量...")
        file_a_embeddings, embedding_model = generate_embeddings(
            file_a_segments,
            model_name=args.model_name  # 推荐支持多语言的模型
        )
        
        # 步骤 4: 构建 FAISS 内积索引
        logging.info("构建 FAISS 内积索引...")
        faiss_index = build_faiss_index_cosine(file_a_embeddings)
        
        # 可选步骤：保存 FAISS 索引
        # logging.info("保存 FAISS 内积索引...")
        # faiss.write_index(faiss_index, args.faiss_index_path)
        
        # 步骤 5: 读取 FileB 的查询条目
        logging.info("读取 Excel 的查询条目...")
        queries, df_b = read_excel_queries(
            args.query_excel_path,
            query_sheet_name=args.query_sheet_name,    # 根据实际情况调整工作表名称
            query_column=args.query_column    # 根据实际情况调整查询列名
        )
        
        # 步骤 6: 检查相关性并生成输出，包括相关段落和相似度
        logging.info("检查相关性并生成输出...")
        df_output = check_relevance_and_output_cosine(
            queries,
            embedding_model,
            faiss_index,
            file_a_segments,
            df_b,
            top_k=args.top_k,         # 检索最相关的 top_k 个段落
            threshold=args.threshold    # 相似度阈值，可根据实际情况调整
        )
        
        # 步骤 7: 清洗 DataFrame 中的非法字符
        logging.info("清洗 DataFrame 中的非法字符...")
        columns_to_clean = ['Related_Segments', 'Similarities']
        for col in columns_to_clean:
            df_output[col] = df_output[col].apply(remove_illegal_characters)
        
        # 步骤 8: 将相关段落和相似度转换为结构化的多行字符串
        logging.info("格式化相关段落和相似度以便在 Excel 中显示...")
        # 已在 check_relevance_and_output_cosine 函数中完成格式化
        
        # 步骤 9: 保存结果到新的 Excel 文件
        logging.info(f"保存结果到 {args.output_path}...")
        df_output.to_excel(args.output_path, index=False)
        logging.info("完成！")
    except Exception as e:
        logging.error(f"在执行过程中发生错误: {e}")

# 确保在脚本作为主程序运行时执行 main 函数
if __name__ == "__main__":
    main()
