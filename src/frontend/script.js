// script.js

document.getElementById('upload-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const statusDiv = document.getElementById('status');
    const resultDiv = document.getElementById('result');
    const downloadLink = document.getElementById('download-link');

    statusDiv.textContent = "正在上传和处理文件...";
    resultDiv.style.display = "none";

    const formData = new FormData();
    const pdfFile = document.getElementById('pdf_file').files[0];
    const excelFile = document.getElementById('excel_file').files[0];
    const sheetName = document.getElementById('sheet_name').value;
    const queryColumn = document.getElementById('query_column').value;
    const modelName = document.getElementById('model_name').value;
    const maxLength = document.getElementById('max_length').value;
    const topK = document.getElementById('top_k').value;
    const threshold = document.getElementById('threshold').value;
    const useExistingIndex = document.getElementById('use_existing_index').checked;

    formData.append('pdf_file', pdfFile);
    formData.append('excel_file', excelFile);
    formData.append('sheet_name', sheetName);
    formData.append('query_column', queryColumn);
    formData.append('model_name', modelName);
    formData.append('max_length', maxLength);
    formData.append('top_k', topK);
    formData.append('threshold', threshold);
    formData.append('use_existing_index', useExistingIndex);

    try {
        const response = await fetch('http://localhost:8000/process/', {
            method: 'POST',
            body: formData
        });

        if (response.status === 200) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            downloadLink.href = url;
            statusDiv.textContent = "处理完成！";
            resultDiv.style.display = "block";
        } else {
            const error = await response.json();
            statusDiv.textContent = `错误: ${error.message}`;
        }
    } catch (error) {
        statusDiv.textContent = `请求失败: ${error}`;
    }
});
