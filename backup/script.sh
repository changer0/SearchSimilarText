
python3 search_similar_text.py --source_pdf_path "FileA.pdf" \
                 --query_excel_path "FileB.xlsx" \
                 --output_path "FileB_with_Relevance.xlsx" \
                 --query_sheet_name "Sheet1" \
                 --query_column "Query" \
                 --spilt_max_length 500 \
                 --model_name "BAAI/bge-large-zh-v1.5" \
                 --top_k 3 \
                 --threshold 0.5
