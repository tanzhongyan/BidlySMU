# **BidlySMU**

**Bidly** is a predictive tool designed to help SMU students navigate the module bidding system with confidence. Leveraging machine learning models, Bidly predicts minimum bid prices, ensuring students can secure their desired modules with ease. Whether you're trying to optimize your e-credits or plan your bidding strategy, Bidly empowers you with data-driven insights.

---

## **1. Implementation Guide**

### **Step 1: Using the Model**
The project is designed to predict bid points with minimal setup. Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tanzhongyan/BidlySMU
   cd BidlySMU
   ```

2. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment**:
   Ensure you're using **Python 3.8.20**. This ensures compatibility with the dependencies.

4. **Run the Model**:
   - Open the Jupyter Notebook for the version you need:
     - **V1_02_catboost.ipynb**: For predictions using OASIS data.
     - **V2_04_Catboost.ipynb**: For predictions using enhanced data with class timings.
   - Follow the notebook cells to preprocess data, train the model, and generate predictions.

### **Step 2: Accessing the Enhanced Data**
- **Scraped Class Timings Data**: Scraping BOSS data for class timings (as done in **V2_01_selenium_BossResults.ipynb**) is **time and resource-intensive**. It involves:
  - Running a Selenium bot to scrape class timings.
  - Processing and merging data into a usable format.

  **Recommendation**: Avoid re-running the scraping bot. If you are an SMU student with a valid academic purpose, you may request access to the pre-scraped dataset from the project author.

  **How to Request**:
  1. Contact me with your request details.
  2. Provide your academic rationale and SMU affiliation.
  3. If approved, you’ll receive access to the processed dataset for your use.

### **Step 3: Customizing the Model**
- Use the preprocessing notebooks (**V1_01_SMU_Bidding_Preprocessing.ipynb**, **V2_03_SMU_Bidding_Preprocessing.ipynb**) to adapt the data for your specific use case.
- **Upcoming Features**: Prediction endpoint and website integration are planned. Look out for updates!

---

## **2. Key Differences Between V1 and V2**

- **V1**: Utilizes readily available data from the OASIS BOSS bidding results.
- **V2**: Enhances V1 by incorporating scraped data from the BOSS website, including class timings, for improved predictions.

---

## **3. Project Overview**

### **V1**
- **Purpose**: Evaluate and compare different machine learning models (CatBoost and DNN) using OASIS data.
- **Key Features**:
  1. CatBoost models explored default, tuned, and cleaned versions.
  2. Neural networks were tested but found less suitable for this tabular data.

### **V2**
- **Purpose**: Improve prediction accuracy using additional features like class timings.
- **Key Features**:
  1. Scraped data enhanced the dataset significantly.
  2. Feature selection was performed to manage diminishing returns.

---

## **4. Key Findings**

### **V1 Summary**
1. CatBoost models perform better for tabular data with categorical variables.
2. Default CatBoost is nearly as effective as tuned models.
3. Removing outliers (e.g., troll bids) has limited impact on performance.
4. Latest academic year data provides the most predictive power.
5. Combining data from all years yields the best results.

### **V2 Enhancements**
1. Adding class timings significantly reduces prediction errors.
2. Additional features lead to diminishing returns beyond a certain point.
3. Adding safety factors or bootstrapping confidence intervals could improve predictions but requires more computation.

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
