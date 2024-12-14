# similarity_searcher.py

# 导入所需的库
import pdfplumber          # 用于从 PDF 中提取文本
import pandas as pd        # 用于处理 Excel 文件
import re                  # 用于正则表达式操作
from sentence_transformers import SentenceTransformer  # 用于生成句子嵌入
import faiss               # 用于高效的向量检索
import numpy as np         # 用于数值计算
import logging             # 用于日志记录

class SimilaritySearcher:
    def __init__(self, source_pdf_path, query_excel_path, output_path,
                 query_sheet_name='Sheet1', query_column='Query',
                 split_max_length=500, model_name='BAAI/bge-large-zh-v1.5',
                 top_k=3, threshold=0.5):
        """
        初始化相似度检索器。

        参数:
        - source_pdf_path (str): 来源 PDF 文件路径。
        - query_excel_path (str): 查询 Excel 文件路径。
        - output_path (str): 输出 Excel 文件路径。
        - query_sheet_name (str): Excel 中查询条目所在的工作表名称。
        - query_column (str): Excel 中查询条目所在的列名。
        - split_max_length (int): 文本分段的最大长度。
        - model_name (str): 用于生成嵌入的预训练模型名称。
        - top_k (int): 每个查询检索的相关段落数量。
        - threshold (float): 判断是否相关的相似度阈值。
        """
        self.source_pdf_path = source_pdf_path
        self.query_excel_path = query_excel_path
        self.output_path = output_path
        self.query_sheet_name = query_sheet_name
        self.query_column = query_column
        self.split_max_length = split_max_length
        self.model_name = model_name
        self.top_k = top_k
        self.threshold = threshold

        # 初始化其他属性
        self.file_a_text = ""
        self.file_a_segments = []
        self.file_a_embeddings = None
        self.embedding_model = None
        self.faiss_index = None
        self.queries = []
        self.df_b = None
        self.df_output = None

    def extract_text_from_pdf(self):
        """从 PDF 文件中提取文本内容。"""
        text = ""
        try:
            with pdfplumber.open(self.source_pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            self.file_a_text = text
            logging.info("成功提取 PDF 文件中的文本内容。")
        except Exception as e:
            logging.error(f"提取 PDF 内容时出错: {e}")
            raise e

    def read_excel_queries(self):
        """读取 Excel 文件中的查询条目。"""
        try:
            self.queries, self.df_b = self._read_excel_queries_internal()
            logging.info(f"成功读取 Excel 文件中的 {len(self.queries)} 个查询条目。")
        except Exception as e:
            logging.error(f"读取 Excel 文件时出错: {e}")
            raise e

    def _read_excel_queries_internal(self):
        """内部方法：读取 Excel 文件中的查询条目。"""
        df = pd.read_excel(self.query_excel_path, sheet_name=self.query_sheet_name)
        queries = df[self.query_column].dropna().tolist()
        return queries, df

    def split_text(self):
        """将文本按指定的最大长度分段。"""
        sentences = re.split(r'(?<=[。！？；])', self.file_a_text)
        current_segment = ""
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= self.split_max_length:
                current_segment += sentence
            else:
                self.file_a_segments.append(current_segment)
                current_segment = sentence
        if current_segment:
            self.file_a_segments.append(current_segment)
        logging.info(f"成功将文本分段为 {len(self.file_a_segments)} 个段落。")

    def normalize_embeddings(self, embeddings):
        """
        将嵌入向量归一化为单位向量。

        参数:
        - embeddings (np.ndarray): 原始嵌入向量。

        返回:
        - normalized_embeddings (np.ndarray): 归一化后的嵌入向量。
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 防止除以零
        normalized_embeddings = embeddings / norms
        logging.info("normalize_embeddings 成功归一化嵌入向量。")
        return normalized_embeddings

    def generate_embeddings(self):
        """生成文本段落的嵌入向量。"""
        try:
            self.file_a_embeddings, self.embedding_model = self._generate_embeddings_internal()
            logging.info("成功生成嵌入向量。")
        except Exception as e:
            logging.error(f"生成嵌入向量时出错: {e}")
            raise e

    def _generate_embeddings_internal(self):
        """内部方法：生成嵌入向量。"""
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(self.file_a_segments, convert_to_tensor=False, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        embeddings = self.normalize_embeddings(embeddings)
        return embeddings, model

    def build_faiss_index_cosine(self):
        """构建 FAISS 内积索引，用于余弦相似度检索。"""
        try:
            dimension = self.file_a_embeddings.shape[1]
            logging.info(f"使用 {dimension} 维度构建 FAISS 内积索引。")
            index = faiss.IndexFlatIP(dimension)  # 使用内积索引
            index.add(self.file_a_embeddings)
            self.faiss_index = index
            logging.info(f"成功构建 FAISS 内积索引，包含 {index.ntotal} 个向量。")
        except Exception as e:
            logging.error(f"构建 FAISS 内积索引时出错: {e}")
            raise e

    def retrieve_cosine_similarity(self, query):
        """
        检索与查询最相关的文本段落，基于余弦相似度。

        参数:
        - query (str): 查询字符串。

        返回:
        - results (list): 包含相关段落和相似度的字典列表。
        """
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding).astype('float32')
            query_embedding = self.normalize_embeddings(query_embedding)
        except Exception as e:
            logging.error(f"生成查询嵌入向量时出错: {e}")
            raise e

        try:
            similarities, indices = self.faiss_index.search(query_embedding, self.top_k)
        except Exception as e:
            logging.error(f"使用 FAISS 进行检索时出错: {e}")
            raise e

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.file_a_segments):
                results.append({
                    'segment': self.file_a_segments[idx],
                    'similarity': float(similarity)  # 转换为float，便于JSON序列化
                })
        return results

    def remove_illegal_characters(self, text):
        """
        移除 Excel 不允许的非法字符。

        参数:
        - text (str): 输入的字符串。

        返回:
        - clean_text (str): 清洗后的字符串。
        """
        if not isinstance(text, str):
            return text
        illegal_chars = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
        clean_text = illegal_chars.sub('', text)
        return clean_text

    def check_relevance_and_output_cosine(self):
        """
        检查每个查询条目在 PDF 中的相关性，并生成输出。

        返回:
        - df_output (pd.DataFrame): 更新后的 DataFrame，包含相关性和相似度信息。
        """
        relevance = []              # 存储每个查询的相关性结果（True/False）
        related_segments = []       # 存储每个查询的相关段落内容
        related_similarities = []   # 存储每个查询的相关段落相似度

        for query in self.queries:
            logging.info(f"正在处理查询: {query}")
            # 检索当前查询最相关的段落
            results = self.retrieve_cosine_similarity(query)
            # 判断是否有任何一个检索结果的相似度超过阈值
            has_relevance = any(res['similarity'] >= self.threshold for res in results)
            relevance.append("是" if has_relevance else "否")

            # 如果有相关性，收集相关段落和相似度
            if has_relevance:
                # 筛选出相似度超过阈值的段落
                relevant_results = [res for res in results if res['similarity'] >= self.threshold]
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
        self.df_b['是否有相关性'] = relevance
        # 将相关段落内容和相似度添加到 DataFrame 中
        self.df_b['相关段落'] = related_segments
        self.df_b['相似度'] = related_similarities

        # 数据清洗：移除非法字符
        logging.info("正在清洗数据中的非法字符...")
        columns_to_clean = ['相关段落', '相似度']
        for col in columns_to_clean:
            self.df_b[col] = self.df_b[col].apply(self.remove_illegal_characters)

        logging.info("完成所有查询的相关性检索和数据清洗。")
        return self.df_b

    def save_output(self):
        """保存结果到新的 Excel 文件。"""
        try:
            self.df_output.to_excel(self.output_path, index=False)
            logging.info(f"成功保存结果到 {self.output_path}")
        except Exception as e:
            logging.error(f"保存结果到 Excel 时出错: {e}")
            raise e

    def run(self):
        """执行完整的流程。"""
        # 提取文本
        self.extract_text_from_pdf()
        # 分割文本
        self.split_text()
        # 生成嵌入向量
        self.generate_embeddings()
        # 构建 FAISS 索引
        self.build_faiss_index_cosine()
        # 读取查询
        self.read_excel_queries()
        # 检索相关性并输出
        self.df_output = self.check_relevance_and_output_cosine()
        # 保存结果
        self.save_output()
        logging.info("完成所有步骤。")
