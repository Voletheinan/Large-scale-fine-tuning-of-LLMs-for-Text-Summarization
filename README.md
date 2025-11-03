# Large-scale Fine-tuning of LLMs for Text Summarization

## Mục tiêu dự án

Dự án này tập trung vào việc fine-tuning mô hình ngôn ngữ lớn (LLM) TinyLLaMA cho bài toán tóm tắt văn bản. Chúng ta sẽ khám phá và so sánh hiệu suất của các phương pháp fine-tuning khác nhau, bao gồm LoRA, QLoRA, Adapter (IA3), và Prompt-tuning. Mục tiêu là:

- Áp dụng và triển khai các kỹ thuật fine-tuning hiệu quả.
- So sánh kết quả về độ chính xác (sử dụng ROUGE metrics), thời gian huấn luyện, tài nguyên GPU tiêu thụ, và chất lượng tóm tắt.
- Trực quan hóa các kết quả để rút ra kết luận sâu sắc.
- Tạo một báo cáo tổng hợp các phát hiện.

## Cấu trúc dự án

```
project/
├── data/
│   ├── raw/                  # Dữ liệu gốc (ví dụ: cnn_dailymail train/val/test.csv)
│   └── processed/            # Dữ liệu đã tiền xử lý
├── models/
│   ├── tinyllama_base/       # Trọng số của mô hình TinyLLaMA gốc
│   ├── finetuned_lora/       # Adapter LoRA đã fine-tuned
│   ├── finetuned_qlora/      # Adapter QLoRA đã fine-tuned
│   ├── finetuned_adapter/    # Adapter IA3 đã fine-tuned
│   ├── finetuned_prompt_tuning/ # Adapter Prompt-tuning đã fine-tuned
│   └── checkpoints/          # Checkpoints trong quá trình huấn luyện
├── src/
│   ├── config.py             # Cấu hình chung và hyperparameters
│   ├── prepare_data.py       # Script chuẩn bị dữ liệu
│   ├── dataset_utils.py      # Tiện ích liên quan đến Dataset và Tokenizer
│   ├── train_lora.py         # Script fine-tune với LoRA
│   ├── train_qlora.py        # Script fine-tune với QLoRA
│   ├── train_adapter.py      # Script fine-tune với Adapter (IA3)
│   ├── train_prompt_tuning.py # Script fine-tune với Prompt-tuning
│   ├── evaluate.py           # Script đánh giá mô hình
│   ├── predict.py            # Script tạo tóm tắt dự đoán
│   ├── metrics_utils.py      # Tiện ích cho các hàm metrics
│   └── visualize.py          # Script tạo biểu đồ và trực quan hóa
├── notebooks/
│   ├── 1_Data_Exploration.ipynb        # Khám phá dữ liệu
│   ├── 2_FineTuning_Results.ipynb      # Trực quan hóa kết quả huấn luyện (loss curves)
│   ├── 3_Evaluation_Comparison.ipynb   # So sánh đánh giá định lượng
│   ├── 4_Sample_Predictions.ipynb      # Hiển thị các ví dụ tóm tắt
│   └── 5_Final_Visualization.ipynb     # Tổng hợp và phân tích cuối cùng
├── report/
│   ├── figures/              # Biểu đồ và hình ảnh cho báo cáo
│   ├── tables/               # Bảng số liệu cho báo cáo
│   └── final_report.pdf      # Báo cáo cuối cùng (placeholder)
└── requirements.txt          # Danh sách các thư viện Python cần thiết
```

## Cài đặt

1.  **Clone the repository** (nếu bạn chưa làm):

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Tạo môi trường ảo và cài đặt dependencies**:

    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Trên Windows
    # source venv/bin/activate  # Trên Linux/macOS
    pip install -r requirements.txt
    pip install accelerate # Đảm bảo cài đặt accelerate riêng biệt cho các tính năng phân phối
    ```

3.  **Tải xuống dataset**:

    Dataset CNN/DailyMail sẽ được tải tự động từ Hugging Face khi bạn chạy các script huấn luyện hoặc notebook. Bạn có thể chạy notebook `notebooks/1_Data_Exploration.ipynb` để khám phá dữ liệu sau khi nó được tải lần đầu tiên.

## Cách sử dụng

Thực hiện các bước sau theo thứ tự để chạy toàn bộ dự án:

1.  **Chuẩn bị dữ liệu**:

    Dataset CNN/DailyMail sẽ được tải tự động từ Hugging Face khi bạn chạy các script huấn luyện hoặc notebook. Bạn có thể chạy notebook `notebooks/1_Data_Exploration.ipynb` để khám phá dữ liệu sau khi nó được tải lần đầu tiên.

2.  **Huấn luyện các mô hình**:

    Chạy từng script huấn luyện riêng biệt cho mỗi phương pháp fine-tuning. Đảm bảo bạn đã kích hoạt môi trường ảo và đang ở thư mục gốc của dự án.

    -   **LoRA Fine-tuning**:

        ```bash
        python src/train_lora.py
        ```

    -   **QLoRA Fine-tuning**:

        ```bash
        python src/train_qlora.py
        ```

    -   **Adapter (IA3) Fine-tuning**:

        ```bash
        python src/train_adapter.py
        ```

    -   **Prompt-tuning**:

        ```bash
        python src/train_prompt_tuning.py
        ```

    Các adapter/mô hình fine-tuned sẽ được lưu vào các thư mục tương ứng trong `models/`.

3.  **Đánh giá mô hình**:

    Chạy script đánh giá để tính toán ROUGE scores và các metrics khác cho tất cả các mô hình đã fine-tuned. Kết quả sẽ được lưu vào `report/tables/evaluation_results.csv`.

    ```bash
    python src/evaluate.py
    ```

4.  **Tạo tóm tắt dự đoán**:

    Bạn có thể sử dụng script `predict.py` để tạo tóm tắt từ một bài viết đầu vào hoặc một mẫu từ tập test.

    -   **Ví dụ với một bài viết cụ thể (sử dụng mô hình LoRA)**:

        ```bash
        python src/predict.py --model_type lora --article_text "Your long article text here..."
        ```

    -   **Ví dụ với mẫu từ tập test (sử dụng mô hình QLoRA)**:

        ```bash
        python src/predict.py --model_type qlora --sample_index 5
        ```

5.  **Trực quan hóa và Phân tích kết quả**:

    Sử dụng các notebook trong thư mục `notebooks/` để khám phá dữ liệu, trực quan hóa kết quả huấn luyện, so sánh các phương pháp, và xem xét các mẫu tóm tắt.

    -   `notebooks/2_FineTuning_Results.ipynb`: Để xem biểu đồ loss của quá trình train.
    -   `notebooks/3_Evaluation_Comparison.ipynb`: Để so sánh định lượng các mô hình (ROUGE, thời gian, tham số).
    -   `notebooks/4_Sample_Predictions.ipynb`: Để so sánh chất lượng tóm tắt giữa các mô hình.
    -   `notebooks/5_Final_Visualization.ipynb`: Tổng hợp các biểu đồ và phân tích cuối cùng cho báo cáo.

## Báo cáo cuối cùng

File `report/final_report.pdf` là nơi bạn sẽ tổng hợp tất cả các phân tích, biểu đồ và kết luận của mình. Tham khảo notebook `5_Final_Visualization.ipynb` để có cấu trúc đề xuất cho báo cáo.


# Large-scale-fine-tuning-of-LLMs-for-Text-Summarization
