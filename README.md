# Voting Patterns Analysis (Israeli Elections)

This project explores voting patterns across the 24th and 25th Knesset elections in Israel using data mining techniques and dimensionality reduction (PCA).

## 📊 Project Objective

To identify shifts and similarities in party support across different polling stations and cities, and to group parties based on voter behavior using Principal Component Analysis (PCA).

## 📁 Files Included

- `PCA1.py` – Main Python script for:
  - Data cleaning
  - Filtering significant parties
  - Normalization (Min-Max & Z-Score)
  - PCA transformation & visualization (2D and 3D)
- `24th by city.csv` / `24th by kalpi.csv` – Results of the 24th election
- `25th by city.csv` / `25th by kalpi.csv` – Results of the 25th election
- `Data mining and ML project.pdf` – Full Hebrew documentation with analysis, charts, and insights

## 🛠️ Tools & Libraries

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 🧠 Key Methods

- **EDA** – Descriptive statistics and correlation matrices
- **PCA** – Applied on normalized data to identify key ideological and demographic dimensions
- **Normalization** – Compared Min-Max and Z-Score methods
- **Cross-election analysis** – Compared party patterns between 24th and 25th Knesset elections

## 📌 Highlights

- Identified ideological axes: Left vs. Right, Religious vs. Secular
- Grouped parties based on voter behavior similarity
- Used Hebrew-specific adjustments for text and labels

## 🔍 Example Visualizations

2D and 3D PCA projections showing clusters of similar parties.

*(Visuals not included in repo, but generated via script and shown in the PDF)*

## 📄 License

This project is for educational purposes. Election data from official sources.
