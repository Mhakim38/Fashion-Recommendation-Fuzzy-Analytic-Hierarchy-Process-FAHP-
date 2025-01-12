# %%
import numpy as np
import pandas as pd
import os as os
import streamlit as stc


# %%
file_path = r'C:\Users\hakim\Fashion Recommendation system FAHP\pairwise_matrix_input.csv'
print(os.path.exists(file_path))

# %% [markdown]
# 1. import lib numpy as *np*
# 2. import pandas numpy as *pd*
# 
# *Checking file path....(cell 2)

# %% [markdown]
# # Preprocess Data (Define Fuzzy)

# %%
import pandas as pd

# Define fuzzy mappings for Price, Comfort, and Color
usage_fuzzy_map = {
    'Casual': (1, 1, 2),
    'Sports': (1, 2, 3),
    'Ethnic': (2, 3, 4),
    'Formal': (3, 4, 5),
    'Smart Casual': (4, 5, 6),
    'Party': (5, 6, 7),
    'Travel': (6, 7, 8),
    'Home': (7, 8, 9)
}

gender_fuzzy_map = {
    "Men": (1, 1, 2),
    "Women": (2, 3, 4),
    "Boys": (2, 3, 4),
    "Girls": (3, 4, 5),
    "Unisex": (4, 5, 6)
}

color_fuzzy_map = {
    'Black': (1, 1, 2),
    'Blue': (2, 3, 4),
    'Brown': (3, 4, 5),
    'Gold': (4, 5, 6),
    'Green': (5, 6, 7),
    'Grey': (6, 7, 8),
    'Khaki': (7, 8, 9),
    'Lavender': (8, 9, 10),
    'Magenta': (9, 10, 11),
    'Maroon': (10, 11, 12),
    'Navy Blue': (11, 12, 13),
    'Orange': (12, 13, 14),
    'Peach': (13, 14, 15),
    'Pink': (14, 15, 16),
    'Purple': (15, 16, 17),
    'Red': (16, 17, 18),
    'Turquoise Blue': (17, 18, 19),
    'White': (18, 19, 20),
    'Yellow': (19, 20, 21)
}

# Define defuzzification function
def defuzzify(fuzzy_weight):
    if isinstance(fuzzy_weight, tuple) and len(fuzzy_weight) == 3:
        L, M, R = fuzzy_weight  # Unpack the fuzzy value (L, M, R)
        crisp_value = (L + 4 * M + R) / 6  # Defuzzification formula
        return crisp_value
    else:
        return None  # Handle the case where fuzzy_weight is not a tuple

# Function to load, process the dataset, and save to a new file
def load_and_process_data(file_path, output_file):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_path)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Count duplicates based on 'id' and 'productDisplayName' before removal
    duplicates_count = df.duplicated(subset=['id', 'productDisplayName']).sum()
    print(f"Number of duplicate rows removed: {duplicates_count}")
    
    # Remove duplicates based on 'id' and 'productDisplayName'
    df = df.drop_duplicates(subset=['id', 'productDisplayName'])
    
    # Apply fuzzy mappings and handle missing values with default tuples
    df['Usage_Fuzzy'] = df['usage'].map(usage_fuzzy_map).apply(lambda x: x if isinstance(x, tuple) else (0, 0, 0))
    df['Gender_Fuzzy'] = df['gender'].map(gender_fuzzy_map).apply(lambda x: x if isinstance(x, tuple) else (0, 0, 0))
    df['Color_Fuzzy'] = df['baseColour'].map(color_fuzzy_map).apply(lambda x: x if isinstance(x, tuple) else (0, 0, 0))
    
    # Apply defuzzification
    df['Usage_Defuzzified'] = df['Usage_Fuzzy'].apply(defuzzify)
    df['Gender_Defuzzified'] = df['Gender_Fuzzy'].apply(defuzzify)
    df['Color_Defuzzified'] = df['Color_Fuzzy'].apply(defuzzify)
    
    # Replace null values with 'NULL' in all columns except the fuzzy-mapped columns
    Replace_Null = df.columns.difference(['Usage_Fuzzy', 'Gender_Fuzzy', 'Color_Fuzzy', 'Usage_Defuzzified', 'Gender_Defuzzified', 'Color_Defuzzified'])
    df[Replace_Null] = df[Replace_Null].fillna('NULL')
    Count_Null = df[Replace_Null].isna().sum().sum()
    print(f"Number of NaN values replaced by 'NULL': {Count_Null}")
    print(" ")

    # Save the processed DataFrame with all original columns plus defuzzified columns
    df.to_csv(output_file, index=False)

    return df

# Example usage
file_path = r'C:\Users\hakim\Fashion Recommendation system FAHP\styles.csv'  # Input file path
output_file = r'C:\Users\hakim\Fashion Recommendation system FAHP\processed_styles_with_defuzz.csv'  # Output file path

processed_data = load_and_process_data(file_path, output_file)

# Display the defuzzified values for each criterion
print(processed_data[['Usage_Fuzzy', 'Usage_Defuzzified', 'Gender_Fuzzy', 'Gender_Defuzzified', 'Color_Fuzzy', 'Color_Defuzzified']])


# %% [markdown]
# 2. This cell is to define those data in the dataset and convert it into Fuzzy number
# 
# ex: Item 1 (Price: RM 50, Style: 7 and comfort: 8)
# * The Style and comfort is a scale from 1-10, so the input must be the same....
# * criteria chosen is price, color, usage (Casual, Formal, sports....), gender

# %% [markdown]
# ## Split Data 80% (training) 20% (Testing)

# %%
# # Step 2: Define Pairwise Comparison Matrix for Criteria (User Input)
# # The user compares each criterion in pairs (Price vs Style, Price vs Comfort, etc.)

# #SPLIT DATA 80% AND 20%
# import random

# # Load the processed data
# file_path = r'C:\Users\hakim\Desktop\UiTM\2. Degree\sem 6\CSP 650\Project Code\Dataset\processed_styles.csv'
# data = pd.read_csv(file_path)

# # Shuffle the data
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # `frac=1` means shuffle all rows

# # Define the split ratio
# split_ratio = 0.8  # 80% training, 20% testing
# split_index = int(len(data) * split_ratio)

# # Split the data manually
# train_data = data[:split_index]
# test_data = data[split_index:]

# # Save the training and test datasets (optional)
# train_data.to_csv('training_data.csv', index=False)
# test_data.to_csv('testing_data.csv', index=False)

# # Print the number of records in each set
# print(f"Training data: {len(train_data)} rows")
# print(f"Testing data: {len(test_data)} rows")


# %% [markdown]
# ## Define Pairwise Comparison "matrix" (For performance evaluation only)

# %% [markdown]
# ![comparison.png](attachment:comparison.png)

# %% [markdown]
# 3. This cell is to get input from user for pairwise comparison between the dataset and the input. 
#     
#     * all data collected needed to convert into Fuzzy number. (Text also need to be convert)
#     * Fuzzy: The comparison between the criteria in needed as it is a form of Fuzzy (Price vs Price, Price vs Style, Price vs Comfort)
#     
#     **This shows on how important those questions to user between the criteria/attribute** 
#     * AHP: Givng the expantion of not using a whole number such as 1-10 and 1.5, 2.8, 5.2 etc... is possible
#     * The image above shows the pairwise comparison between those criteria/attribute (Added color)
# Therefore, Fuzzy Analytic Hierarchy Process (FAHP) is done... 

# %%
# Example Pairwise Comparison Matrix for Usage, Gender, Color for evaluation purpose
print("Comparison Matrix (Hakim)")
pairwise_matrix = np.array([
    [1, 1/5, 1/2],  # Usage vs Usage, Usage vs Gender, Usage vs Color
    [5, 1, 1/10],    # Gender vs Usage, Gender vs Gender, Gender vs Color
    [2, 10, 1]       # Color vs Usage, Color vs Gender, Color vs Color
])

# # Read the CSV file using pandas
# matrix_input = r'C:\Users\hakim\Fashion Recommendation system FAHP\pairwise_matrix_input.csv'
# df = pd.read_csv(matrix_input, header=0, index_col=0)

# # Convert the dataframe to a NumPy array
# array = df.to_numpy()

# Show the matrix
print(pairwise_matrix)

# %% [markdown]
# ### 1. Sum up each column and nomalize it

# %%
# Step 1: Calculate column sums
col_sums = []
for col in range(pairwise_matrix.shape[1]):
    column_sum = 0
    for row in range(pairwise_matrix.shape[0]):
        column_sum += pairwise_matrix[row, col]
    col_sums.append(column_sum)

print("\nColumn Sums:")
print(col_sums)

# Step 2: Normalize the matrix by dividing each element by the column's sum
normalized_matrix = np.zeros_like(pairwise_matrix, dtype=float)
for i in range(pairwise_matrix.shape[0]):  # Loop over rows
    for j in range(pairwise_matrix.shape[1]):  # Loop over columns
        normalized_matrix[i, j] = pairwise_matrix[i, j] / col_sums[j]

print("\nNormalized Matrix:")
print(normalized_matrix)

# %% [markdown]
# ### 2. calculate Fuzzy Weight (row)

# %%
# Step 3: Calculate the fuzzy weights (average of each row)
fuzzy_weights_matrix = []

for i in range(normalized_matrix.shape[0]):
    row_sum = 0
    for j in range(normalized_matrix.shape[1]):
        row_sum += normalized_matrix[i, j]
    average_weight = row_sum / normalized_matrix.shape[1]  
    fuzzy_weights_matrix.append(average_weight)  # Append the weight to the matrix form
    
print("\nFuzzy Weights (Matrix Form):")
print(np.array(fuzzy_weights_matrix))

# Assign weights to variables
fuzzy_usage_weight, fuzzy_gender_weight, fuzzy_color_weight = fuzzy_weights_matrix
print(f"\n1. Usage Weight: {fuzzy_usage_weight}")
print(f"2. Gender Weight: {fuzzy_gender_weight}")
print(f"3. Color Weight: {fuzzy_color_weight}")


# %% [markdown]
# ### 3. Defuzzification (Crisp Value)

# %% [markdown]
# ### Calculate ranking scores (Dataset) and Recommends

# %%
# Defuzzify the weights
defuzzified_usage_weight = defuzzify(fuzzy_usage_weight)
defuzzified_gender_weight = defuzzify(fuzzy_gender_weight)
defuzzified_color_weight = defuzzify(fuzzy_color_weight)

print(f"\ndefuzzified usage weight:{fuzzy_usage_weight}")
print(f"defuzzified gender weight:{fuzzy_gender_weight}")
print(f"defuzzified color weight:{fuzzy_color_weight}")
print("\n")

# Step 4: Calculate the final score for each product and rank them
def calculate_final_scores(df, defuzzified_usage_weight, defuzzified_gender_weight, defuzzified_color_weight):
    # Ensure weights are valid and sum to 1 for proper scoring
    total_weight = defuzzified_usage_weight + defuzzified_gender_weight + defuzzified_color_weight
    if total_weight != 1:
        defuzzified_usage_weight /= total_weight
        defuzzified_gender_weight /= total_weight
        defuzzified_color_weight /= total_weight

    # Calculate the final score as a weighted sum of defuzzified values
    df['Final_Score'] = (
        df['Usage_Defuzzified'] * defuzzified_usage_weight +
        df['Gender_Defuzzified'] * defuzzified_gender_weight +
        df['Color_Defuzzified'] * defuzzified_color_weight
    )

    # Rank products based on their scores (higher score = better rank)
    df['Rank'] = df['Final_Score'].rank(ascending=False)
    
    # Now, add the code for displaying the 10 most recommended items
    top_10_recommended = df[['productDisplayName','Final_Score', 'Rank']].sort_values(by='Final_Score', ascending=False).head(10)

    # Display the top 10 recommended items
    print(top_10_recommended)

    return df

# Apply the scoring function
processed_data = calculate_final_scores(
    processed_data,
    fuzzy_usage_weight,
    fuzzy_gender_weight,
    fuzzy_color_weight
)

# Save the final ranked dataset to a CSV file
final_output_file = r'C:\Users\hakim\Fashion Recommendation system FAHP\ranked_products.csv'
processed_data.to_csv(final_output_file, index=False)

print(f"\nRanked products saved to: {final_output_file}")




# %% [markdown]
# ### 4. Evaluation
# - Using Precision, Recall and F1-Score

# %%
# # Ground truth IDs and predicted top-k IDs
# ground_truth_ids = [40826, 13493, 4602, 38964, 1648, 43120, 43119, 22233, 32608, 57572]
# predicted_top_k = [40826, 13493, 4602, 38964, 1648, 59045, 39487, 38962, 59984, 1565]

# # Convert to sets for easy comparison
# ground_truth_set = set(ground_truth_ids)
# predicted_set = set(predicted_top_k)

# # Calculate True Positives (TP)
# true_positives = ground_truth_set.intersection(predicted_set)
# tp_count = len(true_positives)

# # Calculate False Positives (FP)
# false_positives = predicted_set - ground_truth_set
# fp_count = len(false_positives)

# # Calculate False Negatives (FN)
# false_negatives = ground_truth_set - predicted_set
# fn_count = len(false_negatives)

# # Precision: TP / (TP + FP)
# precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0

# # Recall: TP / (TP + FN)
# recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

# # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
# f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# # Print the components for better understanding
# print("=== Evaluation Details ===")
# print(f"Ground Truth IDs: {ground_truth_ids}")
# print(f"Predicted Top-k IDs: {predicted_top_k}")
# print()
# print("True Positives (TP):", tp_count, true_positives)
# print("False Positives (FP):", fp_count, false_positives)
# print("False Negatives (FN):", fn_count, false_negatives)
# print()
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-Score: {f1_score:.2f}")


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_all_metrics(true_ratings, top_10_recommended, threshold=3.5):
    """
    Calculate MAE, RMSE, Precision, Recall, and F1-score
    
    Parameters:
    true_ratings: actual ratings from users
    predicted_ratings: predicted ratings from the system
    threshold: rating threshold for considering an item relevant (default: 3.5)
    """
    # MAE and RMSE
    mae = mean_absolute_error(true_ratings, top_10_recommended)
    rmse = np.sqrt(mean_squared_error(true_ratings, top_10_recommended))
    
    # Calculate relevant and recommended items
    relevant_items = set(np.where(true_ratings >= threshold)[0])
    recommended_items = set(np.where(top_10_recommended >= threshold)[0])
    
    # True positives
    true_positives = len(relevant_items.intersection(recommended_items))
    
    # Precision
    if len(recommended_items) == 0:
        precision = 0
    else:
        precision = true_positives / len(recommended_items)
    
    # Recall
    if len(relevant_items) == 0:
        recall = 0
    else:
        recall = true_positives / len(relevant_items)
    
    # F1-score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

# Example usage
true_ratings = np.array([4.5, 3.0, 5.0, 2.0, 4.0])
predicted_ratings = np.array([4.2, 3.3, 4.8, 2.1, 3.8])
metrics = calculate_all_metrics(true_ratings, predicted_ratings)
print(metrics)

# %% [markdown]
# 5. This cell is for applying the weight (input) that have been calculated into the dataset

# %%
# # Rank fashion items based on criteria weights
# ranked_items = rank_alternatives(Dataset, weights)
# ranked_items = sorted(ranked_items.items(), key=lambda x: x[1], reverse=True)

# print("\nRanked Items (from best to worst):")
# for item, score in ranked_items:
#     print(f"{item}: {score:.2f}")

# %% [markdown]
# 5. This cell is for the ranking (recommended items) and the output

# %% [markdown]
# Thanks for you time....

# %%



