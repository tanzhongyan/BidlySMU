# **BidlySMU**

**Bidly** is a predictive tool designed to help SMU students navigate the module bidding system with confidence. Leveraging machine learning models, Bidly predicts minimum and medium bid prices, ensuring students can secure their desired modules with ease. Whether you're trying to optimize your e-credits or plan your bidding strategy, Bidly empowers you with data-driven insights.

---

<div style="background-color:#DFFFD6; padding:12px; border-radius:5px; border: 1px solid #228B22;">
  <h2 style="color:#006400;">✅ Looking to Implement This? ✅</h2>
  <p>🚀 <strong>Use the latest model with <a href="example_prediction.ipynb">example_prediction.ipynb</a>!</strong></p>
  <ul>
    <li>📌 **Pre-trained CatBoost model (`.cbm`) included for instant predictions.**</li>
    <li>🔧 **Step-by-step instructions available.**</li>
    <li>⚡ **No re-training required—just load and predict!**</li>
  </ul>
  <p>👉 <a href="example_prediction.ipynb"><strong>Go to Example Prediction Notebook</strong></a></p>
</div>

<br>

<div style="background-color:#FFD700; padding:15px; border-radius:5px; border: 2px solid #FF4500;">
    
  <h2 style="color:#8B0000;">⚠️🚨 SCRAPE THIS DATA AT YOUR OWN RISK 🚨⚠️</h2>
  
  <p><strong>📌 If you need the data, please contact me directly.</strong> Only available for **existing students**.</p>

  <h3>🔗 📩 How to Get the Data?</h3>
  <p>📨 <strong>Reach out to me for access</strong> instead of scraping manually.</p>

</div>

---

## **2. Key Differences Between Versions**
- **V3 (Recommended)**: Introduced bug fixes, two **pre-trained `.cbm` model** for **minimum and median bid predictions** without re-training.
- **V2 (Deprecated)**: Enhances V1 by incorporating scraped data from the BOSS website, including class timings, for improved predictions.
- **V1 (Deprecated)**: Utilizes readily available data from the OASIS BOSS bidding results.

---

## **3. Project Overview**

### **V3 (Recommended)**
- **Purpose**: Provides a pre-trained `.cbm` model for instant predictions without requiring dataset reprocessing.
- **Key Features**:
  1. Directly **loads pre-trained models** from the repository.
  2. No need for additional feature engineering.
  3. Works out-of-the-box with structured course data.

### **V2** (Deprecated)
- **Purpose**: Improve prediction accuracy using additional features like class timings.
- **Key Features**:
  1. Scraped data enhanced the dataset significantly.
  2. Feature selection was performed to manage diminishing returns.

### **V1** (Deprecated)
- **Purpose**: Evaluate and compare different machine learning models (CatBoost and DNN) using OASIS data.
- **Key Features**:
  1. CatBoost models explored default, tuned, and cleaned versions.
  2. Neural networks were tested but found less suitable for this tabular data.
---

## **4. Key Findings**

### **V3 Improvements**
1. No re-training required—models are pre-trained and available as `.cbm` files.
2. Fixed bugs such as dependency on future values and outlier removal.

### **V2 Enhancements**
1. Adding class timings significantly reduces prediction errors.
2. Additional features lead to diminishing returns beyond a certain point.
3. Adding safety factors or bootstrapping confidence intervals could improve predictions but requires more computation.

### **V1 Summary**
1. CatBoost models perform better for tabular data with categorical variables.
2. Default CatBoost is nearly as effective as tuned models.
3. Removing outliers (e.g., troll bids) has limited impact on performance.
4. Latest academic year data provides the most predictive power.
5. Combining data from all years yields the best results.

---

## **5. Crediting the Author**

If you use this project or its models in your work, please credit **Tan Zhong Yan** in the following way on GitHub:

```markdown
Project by [Tan Zhong Yan](https://github.com/tanzhongyan).  
LinkedIn: [Zhong Yan Tan](https://www.linkedin.com/in/zhong-yan-tan/)
```

Feel free to copy and paste the above credit directly into your repository’s README or documentation.

---

## **6. Disclaimer**

The models developed in this project are not guaranteed to be fully accurate. While they can provide valuable insights, they have limitations:
- Predictions can sometimes be unrealistic (e.g., negative values).
- Use these models at your discretion and consider adding safety factors to predictions.

This project is a **work in progress**, and efforts will continue to refine the models for better performance and reliability.