# 导入所需的库
import pdfplumber          # 用于从 PDF 中提取文本
import pandas as pd        # 用于处理 Excel 文件
import re                  # 用于正则表达式操作
from sentence_transformers import SentenceTransformer  # 用于生成句子嵌入
import faiss               # 用于高效的向量检索
import numpy as np         # 用于数值计算

# 函数：从 PDF 文件中提取文本内容
def extract_text_from_pdf(pdf_path):
    """
    从指定的 PDF 文件中提取所有文本内容。

    参数:
    - pdf_path (str): PDF 文件的路径。

    返回:
    - text (str): 提取的完整文本内容。
    """
    text = ""
    # 使用 pdfplumber 打开 PDF 文件
    with pdfplumber.open(pdf_path) as pdf:
        # 遍历每一页
        for page in pdf.pages:
            # 提取当前页的文本
            page_text = page.extract_text()
            if page_text:
                # 将提取的文本添加到总文本中，并添加换行符
                text += page_text + "\n"
    return text

# 函数：读取 Excel 文件中的查询条目
def read_excel_queries(excel_path, sheet_name=0, query_column='Query'):
    """
    从指定的 Excel 文件中读取查询条目。

    参数:
    - excel_path (str): Excel 文件的路径。
    - sheet_name (str 或 int): 要读取的工作表名称或索引。默认值为第一个工作表（0）。
    - query_column (str): 包含查询条目的列名。默认值为 'Query'。

    返回:
    - queries (list): 包含所有查询条目的列表。
    - df (DataFrame): 包含 Excel 文件中所有数据的 Pandas DataFrame。
    """
    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    # 提取指定列中的非空查询条目，并转换为列表
    queries = df[query_column].dropna().tolist()
    return queries, df

# 函数：将文本按指定的最大长度分段
def split_text(text, max_length=500):
    """
    将长文本按最大长度分段，确保句子不被拆分。

    参数:
    - text (str): 要分段的原始文本。
    - max_length (int): 每个段落的最大字符数。默认值为 500。

    返回:
    - segments (list): 分段后的文本列表。
    """
    # 使用正则表达式根据中文句子终结符进行分割
    sentences = re.split(r'(?<=[。！？；])', text)
    segments = []            # 存储分段后的结果
    current_segment = ""     # 当前段落的临时存储
    for sentence in sentences:
        # 检查添加当前句子后是否超过最大长度
        if len(current_segment) + len(sentence) <= max_length:
            current_segment += sentence
        else:
            # 当前段落已满，添加到 segments 列表
            segments.append(current_segment)
            # 开始一个新的段落
            current_segment = sentence
    # 添加最后一个段落（如果存在）
    if current_segment:
        segments.append(current_segment)
    return segments

# 函数：生成文本段落的嵌入向量
def generate_embeddings(segments, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """
    使用预训练的 SentenceTransformer 模型生成文本段落的嵌入向量。

    参数:
    - segments (list): 要生成嵌入的文本段落列表。
    - model_name (str): 预训练模型的名称。默认值为 'paraphrase-multilingual-MiniLM-L12-v2'。

    返回:
    - embeddings (ndarray): 生成的嵌入向量，形状为 (段落数, 向量维度)。
    - model (SentenceTransformer): 加载的 SentenceTransformer 模型。
    """
    # 加载指定的预训练模型
    model = SentenceTransformer(model_name)
    # 生成嵌入向量，设置 convert_to_tensor=False 返回 NumPy 数组
    embeddings = model.encode(segments, convert_to_tensor=False, show_progress_bar=True)
    # 将嵌入转换为 float32 类型的 NumPy 数组（FAISS 需要此类型）
    embeddings = np.array(embeddings).astype('float32')
    return embeddings, model

# 函数：构建 FAISS 向量索引
def build_faiss_index(embeddings, embedding_dim):
    """
    使用 FAISS 构建一个简单的 L2 距离索引。

    参数:
    - embeddings (ndarray): 嵌入向量，形状为 (段落数, 向量维度)。
    - embedding_dim (int): 嵌入向量的维度。

    返回:
    - index (faiss.IndexFlatL2): 构建好的 FAISS 索引。
    """
    # 创建一个 FAISS 索引，使用 L2 距离度量
    index = faiss.IndexFlatL2(embedding_dim)
    # 将所有嵌入向量添加到索引中
    index.add(embeddings)
    return index

# 函数：检索与查询最相关的文本段落
def retrieve_relevant_segments(query, model, index, segments, top_k=1):
    """
    根据查询语句检索最相关的文本段落。

    参数:
    - query (str): 查询语句。
    - model (SentenceTransformer): 用于生成嵌入的模型。
    - index (faiss.IndexFlatL2): FAISS 向量索引。
    - segments (list): 存储所有文本段落的列表。
    - top_k (int): 要检索的最相关段落数量。默认值为 1。

    返回:
    - results (list of dict): 包含检索结果的列表，每个结果包括段落内容、距离和相似度。
    """
    # 生成查询语句的嵌入向量
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')
    # 在 FAISS 索引中搜索最接近的 top_k 个嵌入
    distances, indices = index.search(query_embedding, top_k)
    results = []
    # 遍历每个检索到的结果
    for distance, idx in zip(distances[0], indices[0]):
        results.append({
            'segment': segments[idx],                   # 对应的文本段落
            'distance': distance,                       # L2 距离
            'similarity': 1 / (1 + distance)            # 简单转换为相似度（可调整）
        })
    return results

# 函数：检查每个查询条目在 PDF 中的相关性，并生成输出
def check_relevance_and_output(queries, model, index, segments, df_b, top_k=1, threshold=0.5):
    """
    对每个查询条目进行相关性检索，并在 DataFrame 中标注是否有相关内容。

    参数:
    - queries (list): 查询条目列表。
    - model (SentenceTransformer): 用于生成嵌入的模型。
    - index (faiss.IndexFlatL2): FAISS 向量索引。
    - segments (list): 存储所有文本段落的列表。
    - df_b (DataFrame): 包含查询条目的原始 Excel 数据。
    - top_k (int): 每个查询检索的相关段落数量。默认值为 1。
    - threshold (float): 判断是否相关的相似度阈值。默认值为 0.5。

    返回:
    - df_b (DataFrame): 更新后的 DataFrame，新增 'Has_Relevance_in_FileA' 列。
    """
    relevance = []  # 存储每个查询的相关性结果
    for query in queries:
        # 检索当前查询最相关的段落
        results = retrieve_relevant_segments(query, model, index, segments, top_k=top_k)
        # 判断是否有任何一个检索结果的相似度超过阈值
        has_relevance = any(res['similarity'] >= threshold for res in results)
        relevance.append(has_relevance)
    # 将相关性结果添加到 DataFrame 中
    df_b['Has_Relevance_in_FileA'] = relevance
    return df_b

# 主函数：整合所有步骤，实现完整的流程
def main():
    """
    主函数，执行从 PDF 提取文本、读取 Excel 查询、生成嵌入、构建索引、
    检索相关性并输出结果的完整流程。
    """
    # 定义文件路径
    file_a_path = "FileA.pdf"                    # PDF 文件路径
    file_b_path = "FileB.xlsx"                   # Excel 文件路径
    output_path = "输出结果.xlsx"    # 输出结果的 Excel 文件路径

    # 步骤 1: 提取 FileA 的文本内容
    print("提取 FileA 的文本内容...")
    file_a_text = extract_text_from_pdf(file_a_path)

    # 步骤 2: 分段 FileA 的文本内容
    print("分段 FileA 的文本内容...")
    file_a_segments = split_text(file_a_text, max_length=500)  # 可根据需要调整 max_length

    # 步骤 3: 生成 FileA 的嵌入向量，并加载嵌入模型
    print("生成 FileA 的嵌入向量...")
    file_a_embeddings, embedding_model = generate_embeddings(
        file_a_segments,
        model_name='BAAI/bge-large-zh-v1.5'  # 推荐支持多语言的模型
    )

    # 获取嵌入向量的维度
    embedding_dim = file_a_embeddings.shape[1]

    # 步骤 4: 构建 FAISS 向量索引
    print("构建 FAISS 索引...")
    faiss_index = build_faiss_index(file_a_embeddings, embedding_dim)

    # 步骤 5: 读取 FileB 的查询条目
    print("读取 FileB 的查询条目...")
    queries, df_b = read_excel_queries(
        file_b_path,
        sheet_name="Sheet1",    # 根据实际情况调整工作表名称
        query_column='Query'    # 根据实际情况调整查询列名
    )

    # 步骤 6: 检查相关性并生成输出
    print("检查相关性并生成输出...")
    df_output = check_relevance_and_output(
        queries,
        embedding_model,
        faiss_index,
        file_a_segments,
        df_b,
        top_k=1,         # 检索最相关的 top_k 个段落
        threshold=0.5    # 相似度阈值，可根据实际情况调整
    )

    # 步骤 7: 保存结果到新的 Excel 文件
    print(f"保存结果到 {output_path}...")
    df_output.to_excel(output_path, index=False)
    print("完成！")

# 确保在脚本作为主程序运行时执行 main 函数
if __name__ == "__main__":
    main()
