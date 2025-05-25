import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'cp1255', 'windows-1255', 'iso-8859-8']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, index_col=False, skipinitialspace=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df = df.dropna(axis=1, how='all')
            df.index = df.index + 1
            return df
        except UnicodeDecodeError:
            continue
    return None


base_path = Path(r"C:\workspace\data_mining\PCA_exrecise")

df_24_city = read_csv_with_encoding(base_path / "24th by city.csv")
df_24_kalpi = read_csv_with_encoding(base_path / "24th by kalpi.csv")
df_25_city = read_csv_with_encoding(base_path / "25th by city.csv")
df_25_kalpi = read_csv_with_encoding(base_path / "25th by kalpi.csv")

# 2 a
print(df_24_city.shape)
print(df_24_kalpi.shape)
print(df_25_city.shape)
print(df_25_kalpi.shape)

# 2 b.i
df_25_kalpi = df_25_kalpi[
    ~df_25_kalpi.astype(str).apply(lambda x: x.str.contains('מעטפות חיצוניות', na=False)).any(axis=1)]
print(df_25_kalpi.shape)


# 2 b.ii
def filter_large_parties(df, valid_votes_column_index, party_columns_start_index):
    # חישוב סך הקולות התקפים
    valid_votes_column = df.columns[valid_votes_column_index]
    total_votes = df[valid_votes_column].sum()

    parties_columns = df.columns[party_columns_start_index:]  # עמודות המפלגות החל מהעמודה ה-12
    parties_percent = df[parties_columns].sum() / total_votes * 100

    large_parties = parties_percent[parties_percent > 1].index

    return df.iloc[:, :party_columns_start_index].join(df[large_parties])


df_25_kalpi_filtered = filter_large_parties(df_25_kalpi, valid_votes_column_index=10, party_columns_start_index=11)


# 2.b.iii
party_name_mapping = {
    "אמת": "העבודה",
    "ב": "הבית היהודי",
    "ג": "יהדות התורה",
    "ד": "בלד",
    "ום": "חדש תעל",
    "ט": "הציונות הדתית",
    "כן": "המחנה הממלכתי",
    "ל": "ישראל ביתנו",
    "מחל": "הליכוד",
    "מרצ": "מרצ",
    "עם": "הרשימה הערבית המאוחדת",
    "פה": "יש עתיד",
    "שס": "שס"
}


# פונקציה להחלפת שמות עמודות
def rename_party_columns(df, party_columns_start_index, party_name_mapping):
    # קבלת עמודות המפלגות בלבד
    party_columns = df.columns[party_columns_start_index:]

    # עדכון השמות לפי המילון
    new_columns = {
        col: party_name_mapping.get(col, col) for col in party_columns
    }

    # עדכון שמות העמודות בטבלה
    df.rename(columns=new_columns, inplace=True)
    return df


# החלפת שמות המפלגות בנתוני 25_kalpi
df_25_kalpi_renamed = rename_party_columns(df_25_kalpi_filtered, party_columns_start_index=11,
                                           party_name_mapping=party_name_mapping)
df_25_kalpi = df_25_kalpi_renamed
df_25_kalpi = df_25_kalpi.iloc[:, 11:]

# 2.b.iv
print(df_25_kalpi.shape)
print(df_25_kalpi.dtypes)
# בדיקת כמות הערכים החסרים בכל עמודה
missing_values = df_25_kalpi.isnull().sum()

# הצגת כמות הערכים החסרים לכל עמודה
print("Missing values per column:")
print(missing_values)

# 2.b.v
# הפעלת describe ושמירה כ-DataFrame
describe_df = df_25_kalpi.describe()
# הצגת ה-DataFrame
print(describe_df)

#  הצגת הגרף
sns.pairplot(df_25_kalpi, plot_kws={'s': 1})
plt.show()


# חישוב מטריצת הקורלציה
correlation_matrix = df_25_kalpi.corr()

# הצגת מטריצת הקורלציה עם matshow
plt.matshow(correlation_matrix, cmap="coolwarm")

# הוספת כותרת וצירי X ו-Y
plt.title("Correlation Matrix", pad=20)
plt.colorbar()  # סרגל צבעים
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()


# חזרה על סעיף 2 עם נתוני בחירות 24 ברמת קלפי

def filter_large_parties(df, valid_votes_column_index, party_columns_start_index):
    # חישוב סך הקולות התקפים
    valid_votes_column = df.columns[valid_votes_column_index]
    total_votes = df[valid_votes_column].sum()

    parties_columns = df.columns[party_columns_start_index:]  # עמודות המפלגות החל מהעמודה ה-12
    parties_percent = df[parties_columns].sum() / total_votes * 100

    large_parties = parties_percent[parties_percent > 1].index

    return df.iloc[:, :party_columns_start_index].join(df[large_parties])


df_24_kalpi_filtered = filter_large_parties(df_24_kalpi, valid_votes_column_index=10, party_columns_start_index=11)

party_name_mapping24 = {
    "אמת": "העבודה",
    "ב": "ימינה",
    "ג": "יהדות התורה",
    "ודעם": "הרשימה המשותפת",
    "ט": "הציונות הדתית",
    "כן": "כחול לבן",
    "ל": "ישראל ביתנו",
    "מחל": "הליכוד",
    "מרצ": "מרצ",
    "עם": "הרשימה הערבית המאוחדת",
    "פה": "יש עתיד",
    "שס": "שס",
    "ת": "תקווה חדשה"
}


# פונקציה להחלפת שמות עמודות
def rename_party_columns(df, party_columns_start_index, party_name_mapping24):
    # קבלת עמודות המפלגות בלבד
    party_columns = df.columns[party_columns_start_index:]

    # עדכון השמות לפי המילון
    new_columns = {
        col: party_name_mapping24.get(col, col) for col in party_columns
    }

    # עדכון שמות העמודות בטבלה
    df.rename(columns=new_columns, inplace=True)
    return df

# החלפת שמות המפלגות בנתוני 24_kalpi
df_24_kalpi_renamed = rename_party_columns(df_24_kalpi_filtered, party_columns_start_index=11,
                                           party_name_mapping24=party_name_mapping24)
df_24_kalpi = df_24_kalpi_renamed

df_24_kalpi = df_24_kalpi[
    ~df_24_kalpi.astype(str).apply(lambda x: x.str.contains('מעטפות חיצוניות', na=False)).any(axis=1)]
df_24_kalpi = df_24_kalpi.iloc[:, 11:]

# 2.b.iv
print(df_24_kalpi.shape)
print(df_24_kalpi.dtypes)
# בדיקת כמות הערכים החסרים בכל עמודה
missing_values = df_24_kalpi.isnull().sum()

# הצגת כמות הערכים החסרים לכל עמודה
print("Missing values per column:")
print(missing_values)

# 2.b.v
# הפעלת describe ושמירה כ-DataFrame
describe_df2 = df_24_kalpi.describe()
# הצגת ה-DataFrame
print(describe_df2)

#  הצגת הגרף
sns.pairplot(df_24_kalpi, plot_kws={'s': 1})
plt.show()

# חישוב מטריצת הקורלציה
correlation_matrix = df_24_kalpi.corr()

# הצגת מטריצת הקורלציה עם matshow
plt.matshow(correlation_matrix, cmap="coolwarm")

# הוספת כותרת וצירי X ו-Y
plt.title("Correlation Matrix", pad=20)
plt.colorbar()  # סרגל צבעים
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()


# 3.b

df_transposed = df_25_kalpi.T

# Applying PCA to the transposed DataFrame
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_transposed.values)

# Creating a DataFrame for PCA results
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=df_transposed.index)

# Plotting the PCA results
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(10, 8))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7, color='skyblue', edgecolor='black')

# Adding party names (from the original column names, now indices of the transposed DataFrame)
for party in pca_df.index:
    plt.text(pca_df.loc[party, "PC1"] + 0.1, pca_df.loc[party, "PC2"], party[::-1], fontsize=9)  # הפיכת הטקסט ידנית

# Adding labels, title, and grid
plt.title("PCA of Parties After Transposing the DataFrame", fontsize=16)
plt.xlabel("Principal Component 1 (PC1)", fontsize=14)
plt.ylabel("Principal Component 2 (PC2)", fontsize=14)
plt.grid(True)
plt.show()








# Normalize using Min-Max scaling
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_25_kalpi)

# Convert back to DataFrame for better handling
df_normalized = pd.DataFrame(df_normalized, columns=df_25_kalpi.columns)

# Step 2: Transpose the DataFrame so rows become columns and vice versa
df_transposed = df_normalized.T

# Step 3: Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_transposed.values)

# Step 4: Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=df_transposed.index)

# Step 5: Plot the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7, color='skyblue', edgecolor='black')

# Add labels for each party
for party in pca_df.index:
    plt.text(pca_df.loc[party, "PC1"] + 0.02, pca_df.loc[party, "PC2"], party[::-1], fontsize=9)  # Reverse for Hebrew

# Add labels, title, and grid
plt.title("PCA of Min-Max Normalized Votes Data (Reduced to 2 Dimensions)", fontsize=16)
plt.xlabel("Principal Component 1 (PC1)", fontsize=14)
plt.ylabel("Principal Component 2 (PC2)", fontsize=14)
plt.grid(True)
plt.show()








#  Normalize using Z-Standardization
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_25_kalpi)

# Convert back to DataFrame for better handling
df_standardized = pd.DataFrame(df_standardized, columns=df_25_kalpi.columns)

# Step 2: Transpose the DataFrame so rows become columns and vice versa
df_transposed = df_standardized.T

# Step 3: Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_transposed.values)

# Step 4: Create a DataFrame for PCA results
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=df_transposed.index)

# Step 5: Plot the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7, color='skyblue', edgecolor='black')

# Add labels for each party
for party in pca_df.index:
    plt.text(pca_df.loc[party, "PC1"] + 0.02, pca_df.loc[party, "PC2"], party[::-1], fontsize=9)  # Reverse for Hebrew

# Add labels, title, and grid
plt.title("PCA of Z-Standardized Votes Data (Reduced to 2 Dimensions)", fontsize=16)
plt.xlabel("Principal Component 1 (PC1)", fontsize=14)
plt.ylabel("Principal Component 2 (PC2)", fontsize=14)
plt.grid(True)
plt.show()













scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_25_kalpi)

# חזרה ל-DataFrame עם עמודות זהות לשם של כל מפלגה
df_standardized = pd.DataFrame(df_standardized, columns=df_25_kalpi.columns)

# ביצוע PCA ל-3 מימדים על הנתונים המנורמלים
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_standardized.transpose())

# איור תלת-ממדי
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c='blue', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], party_name[::-1], fontsize=8)

ax.set_title("PCA - 3D Representation")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()



# איור 2D עבור PC1 ו-PC2
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    plt.text(pca_result[i, 0] + 0.2, pca_result[i, 1], party_name[::-1], fontsize=8)

plt.title("PCA - PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# איור 2D עבור PC1 ו-PC3
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 2], c='green', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    plt.text(pca_result[i, 0] + 0.2, pca_result[i, 2], party_name[::-1], fontsize=8)

plt.title("PCA - PC1 vs PC3")
plt.xlabel("PC1")
plt.ylabel("PC3")
plt.grid(True)
plt.show()

# איור 2D עבור PC2 ו-PC3
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 1], pca_result[:, 2], c='red', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    plt.text(pca_result[i, 1] + 0.2, pca_result[i, 2], party_name[::-1], fontsize=8)

plt.title("PCA - PC2 vs PC3")
plt.xlabel("PC2")
plt.ylabel("PC3")
plt.grid(True)
plt.show()











scaler = StandardScaler()
df_standardized = scaler.fit_transform(df_24_kalpi)

# חזרה ל-DataFrame עם עמודות זהות לשם של כל מפלגה
df_standardized = pd.DataFrame(df_standardized, columns=df_24_kalpi.columns)

# ביצוע PCA ל-3 מימדים על הנתונים המנורמלים
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df_standardized.transpose())

# איור תלת-ממדי
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c='blue', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], party_name[::-1], fontsize=8)

ax.set_title("PCA - 3D Representation")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()



# איור 2D עבור PC1 ו-PC2
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    plt.text(pca_result[i, 0] + 0.2, pca_result[i, 1], party_name[::-1], fontsize=8)

plt.title("PCA - PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# איור 2D עבור PC1 ו-PC3
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 2], c='green', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    plt.text(pca_result[i, 0] + 0.2, pca_result[i, 2], party_name[::-1], fontsize=8)

plt.title("PCA - PC1 vs PC3")
plt.xlabel("PC1")
plt.ylabel("PC3")
plt.grid(True)
plt.show()

# איור 2D עבור PC2 ו-PC3
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 1], pca_result[:, 2], c='red', s=50)

# הוספת שמות המפלגות על הגרף
for i, party_name in enumerate(df_standardized.columns):
    plt.text(pca_result[i, 1] + 0.2, pca_result[i, 2], party_name[::-1], fontsize=8)

plt.title("PCA - PC2 vs PC3")
plt.xlabel("PC2")
plt.ylabel("PC3")
plt.grid(True)
plt.show()








def filter_large_parties_by_percentage(df, valid_votes_column, start_party_index, threshold=1):
    # חשב את סך כל הקולות התקפים
    total_valid_votes = df[valid_votes_column].sum()

    # חשב את האחוזים של כל מפלגה
    parties_columns = df.columns[start_party_index:]
    parties_percentages = df[parties_columns].sum() / total_valid_votes * 100

    # שמור רק מפלגות עם יותר מ-1%
    large_parties = parties_percentages[parties_percentages > threshold].index

    # שמור רק עמודות "שם ישוב", "כשרים" והמפלגות הגדולות
    columns_to_keep = ["שם ישוב", valid_votes_column] + list(large_parties)
    filtered_df = df[columns_to_keep]
    return filtered_df

# החלת הפונקציה על שתי הטבלאות
df_25_city_filtered = filter_large_parties_by_percentage(df_25_city, valid_votes_column="כשרים", start_party_index=7)
df_24_city_filtered = filter_large_parties_by_percentage(df_24_city, valid_votes_column="כשרים", start_party_index=7)


common_cities = set(df_25_city_filtered["שם ישוב"]).intersection(set(df_24_city_filtered["שם ישוב"]))

# סינון היישובים המשותפים בשתי הטבלאות
df_25_city_common = df_25_city_filtered[df_25_city_filtered["שם ישוב"].isin(common_cities)]
df_24_city_common = df_24_city_filtered[df_24_city_filtered["שם ישוב"].isin(common_cities)]
print(df_24_city_common.shape)
print(df_25_city_common.shape)






# נבחר רק את העמודות של המפלגות (החל מהעמודה השלישית)
columns_to_rename_25 = df_25_city_common.columns[2:]
columns_to_rename_24 = df_24_city_common.columns[2:]

# נבצע את ההחלפה על עותק של ה-DataFrame
df_25_city_common = df_25_city_common.copy()
df_25_city_common.rename(columns={col: party_name_mapping.get(col, col) for col in columns_to_rename_25}, inplace=True)

df_24_city_common = df_24_city_common.copy()
df_24_city_common.rename(columns={col: party_name_mapping24.get(col, col) for col in columns_to_rename_24}, inplace=True)




# i. יצירת עמודות חדשות בבחירות ה-24
# 1. המחנה הממלכתי = תקווה חדשה + כחול לבן
df_24_city_common["המחנה הממלכתי"] = df_24_city_common["תקווה חדשה"] + df_24_city_common["כחול לבן"]

# 2. חדש-תעל = 0.4 * הרשימה המשותפת
df_24_city_common["חדש תעל"] = 0.4 * df_24_city_common["הרשימה המשותפת"]

# 3. בלד = 0.6 * הרשימה המשותפת
df_24_city_common["בלד"] = 0.6 * df_24_city_common["הרשימה המשותפת"]

# 4. הבית היהודי = 0.4 * ימינה
df_24_city_common["הבית היהודי"] = 0.4 * df_24_city_common["ימינה"]

# 5. הציונות הדתית = 0.6 * ימינה
df_24_city_common["הציונות הדתית"] = 0.6 * df_24_city_common["ימינה"]

# ii. מחיקת העמודות ששימשו בנוסחאות
columns_to_remove = ["תקווה חדשה", "כחול לבן", "הרשימה המשותפת", "ימינה"]
df_24_city_common.drop(columns=columns_to_remove, inplace=True)



# סידור הטבלה של בחירות 24 לפי א'-ב'
df_24_city_common = df_24_city_common.sort_values(by="שם ישוב").reset_index(drop=True)

# סידור הטבלה של בחירות 25 לפי א'-ב'
df_25_city_common = df_25_city_common.sort_values(by="שם ישוב").reset_index(drop=True)

# שמירת שמות היישובים לשימוש עתידי (אם נדרש)
city_names_sorted = df_24_city_common["שם ישוב"].copy()




# שמירת כמות ההצבעות הכשרות ושמות היישובים במשתנים נפרדים
kosher_votes_24 = df_24_city_common["כשרים"].copy()
kosher_votes_25 = df_25_city_common["כשרים"].copy()


# הסרת המשתנים "כשרים" ו"שם ישוב" ממערך הנתונים
df_24_city_common = df_24_city_common.drop(columns=["כשרים", "שם ישוב"])
df_25_city_common = df_25_city_common.drop(columns=["כשרים", "שם ישוב"])






# שלב 1: טרנספוזה ראשונה - להפוך את היישובים לשורות והמפלגות לעמודות
df_24_city_transposed = df_24_city_common.transpose()
df_25_city_transposed = df_25_city_common.transpose()

# שלב 2: תקנון Z-STANDARDIZATION
scaler = StandardScaler()

# תקנון כל משתנה בטבלה של בחירות 24
df_24_city_standardized = pd.DataFrame(
    scaler.fit_transform(df_24_city_transposed),
    index=df_24_city_transposed.index,
    columns=df_24_city_transposed.columns
)

# תקנון כל משתנה בטבלה של בחירות 25
df_25_city_standardized = pd.DataFrame(
    scaler.fit_transform(df_25_city_transposed),
    index=df_25_city_transposed.index,
    columns=df_25_city_transposed.columns
)

# שלב 3: טרנספוזה שנייה - להחזיר את היישובים לשורות ואת המפלגות לעמודות
df_24_city = df_24_city_standardized.transpose()
df_25_city = df_25_city_standardized.transpose()



# ביצוע PCA ל-2 מימדים על הנתונים המתוקננים של בחירות 24
pca_24 = PCA(n_components=2)
pca_24_result = pca_24.fit_transform(df_24_city)

# יצירת DataFrame עם התוצאות
pca_24_df = pd.DataFrame(pca_24_result, columns=["PC1", "PC2"], index=df_24_city.index)




# יצירת DataFrame של ה-components_
loadings = pd.DataFrame(pca_24.components_, columns=df_24_city.columns, index=["PC1", "PC2"]).T

# איור המציג את התרומה של כל מפלגה לצירים החדשים
plt.figure(figsize=(10, 6))
plt.scatter(loadings["PC1"], loadings["PC2"], color='blue', s=100)

# הוספת שמות המפלגות והערכים על הנקודות
for party, row in loadings.iterrows():
    plt.text(row["PC1"] + 0.02, row["PC2"], f"{party[::-1]} ({row['PC1']:.2f}, {row['PC2']:.2f})", fontsize=10)

# כותרות וצירים
plt.title("Contribution of Each Party to PC1 and PC2", fontsize=14)
plt.xlabel("PC1", fontsize=12)
plt.ylabel("PC2", fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()




# רשימת הסדר הרצוי של העמודות
desired_column_order = [
    "העבודה",
    "יהדות התורה",
    "הציונות הדתית",
    "ישראל ביתנו",
    "הליכוד",
    "מרצ",
    "הרשימה הערבית המאוחדת",
    "יש עתיד",
    "שס",
    "המחנה הממלכתי",
    "חדש תעל",
    "בלד",
    "הבית היהודי"
]

# סידור העמודות בטבלה של הכנסת 25 לפי הסדר הרצוי
df_25_city = df_25_city[desired_column_order]





# הטלת נתוני כנסת 24 למערכת הצירים של PCA
projected_24 = pca_24.transform(df_24_city)

# הטלת נתוני כנסת 25 למערכת הצירים של PCA שנמצא מנתוני כנסת 24
projected_25 = pca_24.transform(df_25_city)

# שליפת כמות הקולות הכשרים
valid_votes_24 = kosher_votes_24.values
valid_votes_25 = kosher_votes_25.values

# נירמול כמות הקולות הכשרים עבור הגודל של הנקודות
scaler = MinMaxScaler(feature_range=(10, 100))
size_24 = scaler.fit_transform(valid_votes_24.reshape(-1, 1)).flatten()
size_25 = scaler.transform(valid_votes_25.reshape(-1, 1)).flatten()

# יצירת גרף משולב להצגת שני מערכי הנתונים
plt.figure(figsize=(12, 8))

# הצגת נקודות של נתוני כנסת 24
plt.scatter(
    projected_24[:, 0],
    projected_24[:, 1],
    c='blue',
    label='בחירות כנסת 24',
    alpha=0.6,
    s=size_24
)

# הצגת נקודות של נתוני כנסת 25
plt.scatter(
    projected_25[:, 0],
    projected_25[:, 1],
    c='red',
    label='בחירות כנסת 25',
    alpha=0.6,
    s=size_25
)

# הוספת חצים ליישובים עם לפחות 1000 קולות כשרים
for i, (x1, y1, x2, y2) in enumerate(zip(projected_24[:, 0], projected_24[:, 1], projected_25[:, 0], projected_25[:, 1])):
    if valid_votes_24[i] >= 1000 and valid_votes_25[i] >= 1000:
        plt.arrow(
            x1, y1, x2 - x1, y2 - y1,
            color='gray',
            alpha=0.7,
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True
        )

# כותרות וצירים
plt.title("Projection of Election Data (24th and 25th) onto PCA Axes", fontsize=16)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()