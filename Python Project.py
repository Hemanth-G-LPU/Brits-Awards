# ============================================================
# SECTION 1 — DATA LOADING & INITIAL EXPLORATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("All libraries loaded.")

# Load dataset
df = pd.read_csv(r"C:\Users\heman\Downloads\brit_awards.csv")
print("Dataset loaded successfully.")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# First 5 rows
print(df.head())

# Last 5 rows
print(df.tail())

# Column names
print("Columns:", df.columns.tolist())

# Data types & non-null info
df.info()

# Statistical summary
print(df.describe(include="all"))

# Year range
print(f"Year Range: {df['year'].min()} to {df['year'].max()}")

# Unique categories
print(f"\nUnique Award Categories ({df['details'].nunique()}):")
print(df["details"].value_counts().head(20))


# ============================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Missing values check
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Duplicate rows check
print(f"Duplicate rows: {df.duplicated().sum()}")

# Winners with most awards
print("\nTop 10 Most Awarded Winners:")
print(df["winner"].value_counts().head(10))

# Awards per year
awards_per_year = df.groupby("year").size()
print("\nAwards per year (first 5):")
print(awards_per_year.head())

# Person vs group split
print("\nIs Person (Solo Artist):")
print(df["is_person"].value_counts())

# Most common locations
print("\nTop Venues:")
print(df["location"].value_counts().head(5))

# Summary by category type
cat_summary = df.groupby("details").agg(
    total_awards=("winner", "count"),
    unique_winners=("winner", "nunique"),
    first_year=("year", "min"),
    last_year=("year", "max")
).sort_values("total_awards", ascending=False)
print(cat_summary.head(10))


# ============================================================
# SECTION 3 — DATA CLEANING
# ============================================================

# Step 1 — Standardize column names
df.columns = df.columns.str.strip().str.lower()
print("Columns standardized:", df.columns.tolist())

# Step 2 — Remove duplicate rows
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before - df.shape[0]}")

# Step 3 — Handle missing winner values
print(f"Missing winner values: {df['winner'].isnull().sum()}")
df["winner"] = df["winner"].fillna("Unknown")
print(f"Missing values after fill: {df['winner'].isnull().sum()}")

# Step 4 — Fix data types
df["year"] = df["year"].astype(int)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print(df.dtypes)

# Step 5 — Clean text columns
df["winner"]   = df["winner"].str.strip()
df["details"]  = df["details"].str.strip()
df["location"] = df["location"].str.strip()
print("Text columns cleaned.")

# Step 6 — Create decade column for grouping
df["decade"] = (df["year"] // 10) * 10
print("Decade column created:")
print(df["decade"].value_counts().sort_index())

# Step 7 — Create win_count feature (how many times each winner has won)
win_counts = df["winner"].value_counts().to_dict()
df["winner_total_wins"] = df["winner"].map(win_counts)
print("Winner total wins column created.")

# Step 8 — Reset index
df.reset_index(drop=True, inplace=True)
print(f"Final dataset shape: {df.shape}")
print("Data cleaning complete ")


# ============================================================
# SECTION 4 — VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid")

# --- Chart 1: Awards Given Per Year ---
fig, ax = plt.subplots(figsize=(14, 5))
awards_per_year = df.groupby("year").size()
ax.plot(awards_per_year.index, awards_per_year.values,
        color="royalblue", linewidth=2, marker="o", markersize=4)
ax.fill_between(awards_per_year.index, awards_per_year.values,
                alpha=0.2, color="royalblue")
ax.set_title("Number of BRIT Awards Given Per Year (1982–2025)", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Awards")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("chart1_awards_per_year.png", dpi=150)
plt.show()
'''
# Observation: Awards count varies each year. Late 1980s and early 1990s had
# more categories. Count stabilised in the 2000s and has remained consistent.

# --- Chart 2: Top 15 Most Awarded Winners ---
fig, ax = plt.subplots(figsize=(12, 6))
top_winners = df["winner"].value_counts().head(15)
sns.barplot(x=top_winners.values, y=top_winners.index,
            hue=top_winners.index, palette="viridis", legend=False, ax=ax)
ax.set_title("Top 15 Most Awarded Winners at the BRITs", fontsize=14)
ax.set_xlabel("Total Awards Won")
ax.set_ylabel("Winner")
plt.tight_layout()
plt.savefig("chart2_top_winners.png", dpi=150)
plt.show()

# Observation: Robbie Williams leads with 12 awards. Coldplay, Adele, and
# Take That follow. Solo artists and groups are both well represented.

# --- Chart 3: Solo Artists vs Groups Over the Decades ---
fig, ax = plt.subplots(figsize=(12, 5))
decade_person = df.groupby(["decade", "is_person"]).size().unstack(fill_value=0)
decade_person.plot(kind="bar", stacked=True, colormap="Set2", ax=ax)
ax.set_title("Solo Artists vs Groups Winning Per Decade", fontsize=14)
ax.set_xlabel("Decade")
ax.set_ylabel("Number of Awards")
ax.legend(["Group", "Solo Artist"], title="Type")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("chart3_solo_vs_group_decade.png", dpi=150)
plt.show()

# Observation: Both solo artists and groups win consistently across all
# decades. Groups slightly dominated in the 1980s–1990s era.

# --- Chart 4: Top 10 Award Categories ---
fig, ax = plt.subplots(figsize=(12, 6))
top_cats = df["details"].value_counts().head(10)
sns.barplot(x=top_cats.values, y=top_cats.index,
            hue=top_cats.index, palette="coolwarm", legend=False, ax=ax)
ax.set_title("Top 10 Most Frequent Award Categories", fontsize=14)
ax.set_xlabel("Times Awarded")
ax.set_ylabel("Category")
plt.tight_layout()
plt.savefig("chart4_top_categories.png", dpi=150)
plt.show()

# Observation: Outstanding Contribution has been given the most times (28),
# reflecting the long history of the awards. Core categories like Best
# British Single, Male, Female and Group each appear ~22 times.

# --- Chart 5: Top 10 Venues ---
fig, ax = plt.subplots(figsize=(10, 5))
top_venues = df["location"].value_counts().head(10)
sns.barplot(x=top_venues.values, y=top_venues.index,
            hue=top_venues.index, palette="Blues_d", legend=False, ax=ax)
ax.set_title("Top 10 Venues Hosting the BRIT Awards", fontsize=14)
ax.set_xlabel("Number of Awards Given")
ax.set_ylabel("Venue")
plt.tight_layout()
plt.savefig("chart5_top_venues.png", dpi=150)
plt.show()

# Observation: The O2 arena is the dominant venue (138 awards),
# followed by Earls Court 2 and Earls Court.

# --- Chart 6: Distribution of Winner Total Wins ---
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["winner_total_wins"], bins=20,
             color="steelblue", kde=True, ax=ax)
ax.set_title("Distribution of Total Wins Per Winner", fontsize=14)
ax.set_xlabel("Total Wins")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("chart6_wins_distribution.png", dpi=150)
plt.show()

# Observation: Heavily right-skewed — most winners win only 1–2 times.
# Very few artists win 8+ times, confirming that repeat winners are rare.

# --- Chart 7: Albums/Singles vs Other Awards Over Time ---
fig, ax = plt.subplots(figsize=(14, 5))
album_trend = df.groupby(["year", "is_album_or_single"]).size().unstack(fill_value=0)
album_trend.plot(ax=ax, linewidth=2)
ax.set_title("Album/Single Awards vs Other Awards Per Year", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Awards")
ax.legend(["Other", "Album or Single"], title="Category Type")
plt.tight_layout()
plt.savefig("chart7_album_vs_other.png", dpi=150)
plt.show()

# Observation: Album/Single awards have remained steady over time while
# other category counts fluctuate more noticeably year to year.
'''
# --- Chart 8: Boxplot — Wins Distribution by Is_Person ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x="is_person", y="winner_total_wins", data=df,
            hue="is_person", palette="Set3", legend=False, ax=ax)
ax.set_title("Total Wins Distribution: Solo Artists vs Groups", fontsize=14)
ax.set_xlabel("Artist Type")
ax.set_ylabel("Total Career Wins at BRITs")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Group / Band", "Solo Artist"])
plt.tight_layout()
plt.savefig("chart8_boxplot_wins.png", dpi=150)
plt.show()

# Observation: Solo artists tend to have higher total career wins at the
# BRITs compared to groups, with more high-end outliers (e.g. Robbie Williams).

# --- Chart 9: Scatter Plot — Year vs Winner Total Wins ---
fig, ax = plt.subplots(figsize=(12, 6))
colors = df["is_person"].map({True: "royalblue", False: "tomato"})
ax.scatter(df["year"], df["winner_total_wins"],
           c=colors, alpha=0.5, edgecolors="none", s=40)
ax.set_title("Year vs Winner Total Wins (Solo Artists vs Groups)", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Winner Total Wins")
ax.grid(True, linestyle="--", alpha=0.4)
# Add a manual legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="royalblue",
           markersize=8, label="Solo Artist"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
           markersize=8, label="Group / Band")
]
ax.legend(handles=legend_elements, title="Artist Type")
plt.tight_layout()
plt.savefig("chart9_scatter_year_wins.png", dpi=150)
plt.show()

# Observation: High total-win artists (e.g. Robbie Williams with 12) appear
# mostly in the late 1990s–2000s era. More recent years show lower win totals
# as the dataset is still growing, and repeat winners are rarer post-2010.


# ============================================================
# SECTION 5 — FEATURE ENGINEERING (for ML)
# ============================================================

from sklearn.preprocessing import LabelEncoder

df_ml = df.copy()

# Encode location
le_loc = LabelEncoder()
df_ml["location_encoded"] = le_loc.fit_transform(df_ml["location"])

# Encode details (award category)
le_det = LabelEncoder()
df_ml["details_encoded"] = le_det.fit_transform(df_ml["details"])

# is_album_or_single as int
df_ml["is_album_int"] = df_ml["is_album_or_single"].astype(int)

# Target: is_person (1 = solo artist, 0 = group)
df_ml["target"] = df_ml["is_person"].astype(int)

print("Features ready:")
print(df_ml[["year", "decade", "winner_total_wins",
             "location_encoded", "details_encoded",
             "is_album_int", "target"]].head())


# ============================================================
# SECTION 6 — ML MODEL 1 — LOGISTIC REGRESSION
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Features and target
X = df_ml[["year", "decade", "winner_total_wins",
           "location_encoded", "details_encoded", "is_album_int"]]
y = df_ml["target"]

print(f"Features shape: {X.shape}")
print(f"Target shape  : {y.shape}")
print(f"Class balance — 0 (Group): {(y==0).sum()}  |  1 (Solo): {(y==1).sum()}")

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training size: {X_train.shape}")
print(f"Testing size : {X_test.shape}")

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("Logistic Regression trained successfully!")

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\nLogistic Regression Accuracy: {round(acc_lr * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr,
                             target_names=["Group", "Solo Artist"]))

# Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 5))
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Group", "Solo Artist"],
            yticklabels=["Group", "Solo Artist"], ax=ax)
ax.set_title("Confusion Matrix — Logistic Regression", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("chart10_confusion_lr.png", dpi=150)
plt.show()


# ============================================================
# SECTION 7 — ML MODEL 2 — RANDOM FOREST CLASSIFIER
# ============================================================

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest trained successfully!")

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nRandom Forest Accuracy: {round(acc_rf * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf,
                             target_names=["Group", "Solo Artist"]))

# Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 5))
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Group", "Solo Artist"],
            yticklabels=["Group", "Solo Artist"], ax=ax)
ax.set_title("Confusion Matrix — Random Forest", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("chart11_confusion_rf.png", dpi=150)
plt.show()

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(9, 5))
feature_names = ["Year", "Decade", "Winner Total Wins",
                 "Location", "Award Category", "Is Album/Single"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices],
            y=[feature_names[i] for i in indices],
            hue=[feature_names[i] for i in indices],
            palette="magma", legend=False, ax=ax)
ax.set_title("Feature Importance — Random Forest", fontsize=13)
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("chart12_feature_importance.png", dpi=150)
plt.show()


# ============================================================
# SECTION 8 — MODEL COMPARISON & CONCLUSION
# ============================================================

print("=" * 55)
print("           MODEL COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Metric':<25} {'Logistic Reg':>15} {'Random Forest':>15}")
print("-" * 55)
print(f"{'Accuracy':<25} {round(acc_lr*100,2):>14}% {round(acc_rf*100,2):>14}%")
print("=" * 55)

# Side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title, cmap in zip(
        axes,
        [cm_lr, cm_rf],
        ["Logistic Regression", "Random Forest"],
        ["Blues", "Greens"]):
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["Group", "Solo"], yticklabels=["Group", "Solo"], ax=ax)
    ax.set_title(f"Confusion Matrix — {title}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("chart13_model_comparison.png", dpi=150)
plt.show()

