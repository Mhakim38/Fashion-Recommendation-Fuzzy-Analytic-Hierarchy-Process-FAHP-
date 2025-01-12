from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables for user inputs
user_inputs = {
    'usage_gender': None,
    'usage_color': None,
    'gender_color': None,
    'color': None,
    'usage': None,
    'gender': None,
    'style': None, 
    'occasion': None,
    'season': None,
}

@app.route('/submit_inputs', methods=['POST'])
def submit_inputs():
    global user_inputs
    updated_values = {}

    # Load the dataset
    dataset_path = "C:/Users/hakim/Fashion Recommendation system FAHP/processed_styles_with_defuzz.csv"
    try:
        processed_data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        return jsonify({"message": "Dataset file not found", "path": dataset_path}), 500

    # Process each input from the form
    for key in user_inputs.keys():
        if key in request.form:
            try:
                if key in ['usage_gender', 'usage_color', 'gender_color']:
                    user_inputs[key] = float(request.form[key])  # Convert numeric inputs to float
                else:
                    user_inputs[key] = request.form[key]  # Keep text inputs as string
                updated_values[key] = user_inputs[key]
            except ValueError:
                user_inputs[key] = None
                updated_values[key] = None

    # Debug: Log updated values
    print("Updated Values:", updated_values)

    try:
        # Initialize the pairwise comparison matrix
        usage_gender = user_inputs.get('usage_gender', 1.0)
        usage_color = user_inputs.get('usage_color', 1.0)
        gender_color = user_inputs.get('gender_color', 1.0)

        pairwise_matrix = np.array([
            [1.0, 1 / usage_gender, 1 / usage_color],
            [usage_gender, 1.0, 1 / gender_color],
            [usage_color, gender_color, 1.0]
        ])

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

        # %%
        # step 4 Defuzzify the weights
        def defuzzify(fuzzy_weight):
            if isinstance(fuzzy_weight, tuple) and len(fuzzy_weight) == 3:
                L, M, R = fuzzy_weight  # Unpack the fuzzy value (L, M, R)
                crisp_value = (L + 4 * M + R) / 6  # Defuzzification formula
                return crisp_value
            else:
                return None  # Handle the case where fuzzy_weight is not a tuple

        defuzzified_usage_weight = defuzzify(fuzzy_usage_weight)
        defuzzified_gender_weight = defuzzify(fuzzy_gender_weight)
        defuzzified_color_weight = defuzzify(fuzzy_color_weight)
        
        print(f"\ndefuzzified usage weight:{defuzzified_usage_weight}")
        print(f"defuzzified gender weight:{defuzzified_gender_weight}")
        print(f"defuzzified color weight:{defuzzified_color_weight}")

        # Step 5: Calculate the final score for each product and rank them
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

            df['Rank'] = df['Final_Score'].rank(ascending=False)

            top_10_recommended = df[['productDisplayName', 'Final_Score', 'Rank']].sort_values(by='Final_Score', ascending=False).head(10)
            print("Top 10 Recommendations:")
            print(top_10_recommended)

            return top_10_recommended

        # Apply the scoring function
        top_10_recommended = calculate_final_scores(
            processed_data,
            fuzzy_usage_weight,
            fuzzy_gender_weight,
            fuzzy_color_weight
        )

        # Save ranked products to a CSV
        output_file = r'C:/Users/hakim/Fashion Recommendation system FAHP/ranked_products.csv'
        processed_data.to_csv(output_file, index=False)
        print(f"Ranked products saved to: {output_file}")

        # Return the top recommendations
        recommendations = top_10_recommended.to_dict(orient='records')
        return jsonify({
            "message": "Inputs saved and matrix processed successfully",
            "updated_values": updated_values,
            "all_inputs": user_inputs,
            "received_form_data": dict(request.form),
            "normalized_matrix": normalized_matrix.tolist(),
            "top_10_recommendations": top_10_recommended.to_dict(orient='records')  # Send recommendations
        })

    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({
            "message": "Error during processing",
            "error": str(e)
        }), 500

@app.route('/')
def home():
    return render_template('interface.html')

if __name__ == '__main__':
    app.run(debug=True)
