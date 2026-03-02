# Digital Image Processing

本專案包含「數位影像處理 (Digital Image Processing)」課程的完整實驗實作。

本課程主要探討數位影像處理的核心技術，從空間域的直方圖增強、局部濾波，到頻域的週期性雜訊去除、退化影像復原，以及彩色影像的邊緣偵測與形態學應用。所有實作皆使用 Python 進行開發，並透過嚴謹的數學建模與矩陣運算來驗證影像處理演算法的效果。

## 開發環境與工具

* **語言:** Python 3.x
* **核心套件:** OpenCV, NumPy
* **視覺化工具:** Matplotlib, Pillow (PIL)
* **建置與執行:** Command Line / Python Scripts

## 實驗內容詳細說明

### Lab 1: Histogram Modeling

本實驗目標為分析原始影像的像素分佈，並實作全域與特定化的直方圖轉換技術，以改善空拍圖 (aerial view) 的對比度。

**實作重點:**
* **Histogram Equalization:** 將像素分佈強制轉換為均勻分佈，最大化全域對比度與資訊熵。
* **Histogram Matching:** 依據特定的 Power-Law Distribution 重新映射像素，以精準增強特定灰階區段的細節。

### Lab 2: Hidden Object Extraction

本實驗利用像素的統計特性，顯現低對比度環境下肉眼難以察覺的影像細節。

**實作重點:**
* **Local Enhancement:** 透過計算局部空間視窗內的 Mean 與 Standard Deviation，動態調整像素強度以凸顯隱藏特徵。

### Lab 3: Shading Correction

本實驗目標為設計平滑化濾波器，消除棋盤圖與一般影像中不均勻的光照與網格陰影。

**實作重點:**
* **自定義 Gaussian Lowpass Filter:** 自行設計並實作不同尺寸與標準差的高斯核，精確提取低頻背景層以進行影像光照補償。

### Lab 4: Frequency Domain Denoising

本實驗將影像轉換至頻域，針對正弦波干擾與 Moire Pattern 進行精確的頻譜濾除。

**實作重點:**
* **Auto Peak Detection:** 實作自動化演算法，尋找並鎖定 FFT 頻譜圖中的異常雜訊亮點。
* **Gaussian Notch Filter:** 針對偵測到的雜訊頻率座標，設計高斯帶阻遮罩進行消除，有效避免傳統濾波器的振鈴效應 (Ringing Effect)。

### Lab 5: Image Restoration

本實驗針對大氣湍流與運動模糊進行數學退化建模，並嘗試將嚴重損壞的影像還原。

**實作重點:**
* **Inverse Filtering:** 實作全域與徑向限制的逆向濾波，並處理分母趨近於零時導致的高頻雜訊放大問題。
* **Wiener Filtering:** 引入信噪比參數 (K 值) 進行演算法最佳化平衡，實現更穩定且清晰的影像去模糊 (Deconvolution) 效果。

### Lab 6: Color Image Edge Detection

本實驗在 RGB 彩色向量空間中進行梯度運算，克服傳統轉灰階作法容易遺失色彩邊界細節的缺陷。

**實作重點:**
* **Vector Space Gradient:** 將 Sobel 算子獨立應用於 R、G、B 三個通道，並透過向量幾何計算合成最終的高解析度邊緣強度。

### Lab 7: Morphological Processing

本實驗利用形態學運算處理多種受到干擾的文字影像，並修復斷裂的筆畫以利後續的文字辨識。

**實作重點:**
* **文字修復與增強:** 結合 Dilation、Erosion 與 Closing 等核心技術，去除斑點與正弦波陰影並完美連接斷裂字體。

### Final Project: Video Object Detection

本專案為影像處理課程的期末專題。有別於傳統空間域或頻域的影像處理，本專題導入深度學習技術，針對連續動態影片（`person_dog.mp4`）進行特定物件（人與狗）的精準偵測、計數與追蹤。

**實作重點:**
* **YOLOv8 深度學習模型應用:** 採用 `yolov8x.pt` 預訓練模型為基礎，鎖定特定類別（Class 0: Person, Class 16: Dog）進行高精確度的特徵提取與辨識。
* **即時動態資訊可視化:** 透過 OpenCV 在影片的每一幀中即時繪製 Bounding Boxes，並動態統計當前畫面中的物件總數，同時嵌入專屬識別學號。
* **物件追蹤與效能優化:** 導入 ByteTrack 追蹤演算法搭配自定義的 Confidence 與 IoU 閾值設定（CONF=0.35, IOU=0.55），提升連續影格間的偵測穩定性。
* **Baseline 突破與成效分析:** 成功改善了 Baseline 模型在部分影格中發生的「漏判」缺陷（尤其是針對受到遮蔽或姿態變化的狗）。優化後的模型在嚴苛畫面下依然能保持極高的辨識率 (Confidence > 0.9)。

## 專案結構

```text
Digital-image-processing/
├── lab1/               # Histogram Modeling
├── lab2/               # Hidden Object Extraction
├── lab3/               # Shading Correction
├── lab4/               # Frequency Domain Denoising
├── lab5/               # Image Restoration
├── lab6/               # Color Image Edge Detection
├── lab7/               # Morphological Processing
└── Final_Project/      # YOLOv8 Video Object Detection
