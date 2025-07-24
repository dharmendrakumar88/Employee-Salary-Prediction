import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np # Import numpy for isnan check

# Load the trained model, scaler, encoders, and feature column list
try:
    model = joblib.load("/content/best_model.pkl")
    scaler = joblib.load("/content/scaler.pkl")
    encoders = joblib.load("/content/encoders.pkl")
    feature_columns = joblib.load("/content/feature_columns.pkl")

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}. Please ensure 'best_model.pkl', 'scaler.pkl', 'encoders.pkl', and 'feature_columns.pkl' are in the /content directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs - dynamically create inputs based on the expected feature columns
st.sidebar.header("Input Employee Details")

input_data = {}
for col in feature_columns:
    # Determine input type based on column name or expected data type
    # These should match the selected features and their types
    if col == 'age':
        input_data[col] = st.sidebar.slider("Age", 17, 90, 35) # Example range
    elif col == 'hours-per-week':
         input_data[col] = st.sidebar.slider("Hours per week", 1, 99, 40) # Example range
    elif col == 'capital-gain':
         input_data[col] = st.sidebar.number_input("Capital Gain", 0, 100000, 0) # Example range
    elif col == 'capital-loss':
         input_data[col] = st.sidebar.number_input("Capital Loss", 0, 4500, 0) # Example range
    elif col in encoders and col != 'income': # Categorical columns with loaded encoders (education, occupation, gender, race, native-country)
        if col in encoders:
            original_labels = encoders[col].classes_.tolist()
            selected_value = st.sidebar.selectbox(col.replace('-', ' ').title(), original_labels)
            input_data[col] = selected_value # Store original value
        else:
             st.warning(f"Encoder not found for column: {col}")
             input_data[col] = None # Handle columns without encoders

    else:
        # Default input for any unexpected columns - should not happen with dynamic sidebar
        st.warning(f"Unhandled feature in sidebar generation: {col}")
        input_data[col] = "" # Or some other default/error handling


# Convert input_data to DataFrame
input_df = pd.DataFrame([input_data])

# --- Preprocess the single input data ---

# Create a copy to avoid modifying the original input_df directly
processed_input_df = input_df.copy()

# Apply label encoding to categorical columns using loaded encoders
for col in feature_columns: # Iterate through expected feature columns
     if col in encoders and col != 'income': # Check if it's a categorical column that needs encoding
        if col in processed_input_df.columns: # Ensure column exists in input data
            # Handle unseen labels during single prediction
            if processed_input_df[col][0] in encoders[col].classes_:
                 processed_input_df[col] = encoders[col].transform(processed_input_df[col])
            else:
                 st.warning(f"Selected value '{processed_input_df[col][0]}' for '{col}' not seen in training data. Cannot encode. Setting to NaN.")
                 processed_input_df[col] = np.nan # Use numpy.nan instead of None
        else:
            st.warning(f"Categorical feature '{col}' expected but not found in input data.")
            processed_input_df[col] = np.nan # Add missing categorical column as NaN


# Ensure processed_input_df has the same columns and order as feature_columns
# This step is crucial if the input DataFrame might be missing columns
for col in feature_columns:
    if col not in processed_input_df.columns:
        # Add missing numerical columns with a default (e.g., 0 or mean from training)
        # For simplicity, using 0 here, but consider using the mean/median from training data
        if col in ['age', 'hours-per-week', 'capital-gain', 'capital-loss']:
             processed_input_df[col] = 0
        # Missing categorical columns were handled above by adding np.nan

# Ensure column order matches
processed_input_df = processed_input_df[feature_columns]

# Fill any remaining NaNs that might have been introduced during processing (e.g., from unseen categories)
processed_input_df.fillna(0, inplace=True) # Simple fill with 0, adjust as needed


# Apply scaling to the processed input DataFrame using the loaded scaler
try:
    # Ensure the input DataFrame has the correct numerical columns for scaling
    numerical_cols_to_scale = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
    input_scaled_numerical = scaler.transform(processed_input_df[numerical_cols_to_scale])
    # Replace original numerical columns with scaled ones
    processed_input_df[numerical_cols_to_scale] = input_scaled_numerical

    input_scaled_df = processed_input_df.copy() # Renaming for clarity


except Exception as e:
    st.error(f"Error during scaling single input data: {e}")
    st.stop()


st.write("### 🔎 Processed Input Data (after encoding and scaling)")
st.write(input_scaled_df)

# Predict button
if st.button("Predict Salary Class"):
    # Make prediction using the scaled input data
    try:
        prediction_encoded = model.predict(input_scaled_df)

        # Decode the prediction using the income encoder
        if 'income' in encoders:
            prediction = encoders['income'].inverse_transform(prediction_encoded)
            st.success(f"✅ Predicted Salary Class: {prediction[0]}")
        else:
            st.warning("Income encoder not found. Displaying raw prediction.")
            st.success(f"✅ Raw Prediction: {prediction_encoded[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")


# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())

        # --- Preprocess the batch data ---
        # Apply the same preprocessing steps as the training data

        # Create a copy to avoid modifying the original batch_data DataFrame
        processed_batch_data = batch_data.copy()

        # Handle missing values ('?') - ensure these columns exist in the batch data before processing
        if 'workclass' in processed_batch_data.columns:
             processed_batch_data['workclass'].replace({'?':"Other"},inplace=True)
        if 'occupation' in processed_batch_data.columns:
             processed_batch_data['occupation'].replace({'?':"Other"},inplace=True)
        if 'native-country' in processed_batch_data.columns:
             processed_batch_data["native-country"].replace({'?':"Other"},inplace=True)

        # Drop rows based on conditions (mirroring training) - ensure these columns exist in the batch data
        initial_batch_rows = len(processed_batch_data)
        # Check if columns exist before filtering
        if 'workclass' in processed_batch_data.columns:
             processed_batch_data = processed_batch_data[processed_batch_data['workclass']!='Without-pay']
             processed_batch_data = processed_batch_data[processed_batch_data['workclass']!='Never-worked']
        if 'education' in processed_batch_data.columns: # Keep 'education' in batch processing as it's now a feature
             processed_batch_data = processed_batch_data[processed_batch_data['education']!='5th-6th']
             processed_batch_data = processed_batch_data[processed_batch_data['education']!='1st-4th']
             processed_batch_data = processed_batch_data[processed_batch_data['education']!='Preschool']
        if 'occupation' in processed_batch_data.columns:
             processed_batch_data = processed_batch_data[processed_batch_data['occupation']!='Armed-Forces']
        if 'native-country' in processed_batch_data.columns:
            processed_batch_data = processed_batch_data[processed_batch_data['native-country']!='Holand-Netherlands']

        if len(processed_batch_data) < initial_batch_rows:
            st.warning(f"Filtered out {initial_batch_rows - len(processed_batch_data)} rows during preprocessing.")


        # Drop columns that are NOT in the desired feature list, but ARE in the original batch data
        columns_to_drop = [col for col in processed_batch_data.columns if col not in feature_columns and col != 'income']
        if columns_to_drop:
             processed_batch_data.drop(columns=columns_to_drop, inplace=True)


        # Drop the original 'income' column if it exists in the batch data
        if 'income' in processed_batch_data.columns:
            processed_batch_data.drop(columns=['income'], inplace=True)


        # Ensure processed_batch_data has the same columns as feature_columns.
        # Add missing columns with a default value (e.g., 0) and reindex.
        for col in feature_columns:
            if col not in processed_batch_data.columns:
                st.warning(f"Feature '{col}' expected but not found in batch data. Adding with default value 0.")
                # Decide on default value based on type - using 0 for simplicity
                processed_batch_data[col] = 0


        # Apply label encoding to categorical columns in batch data using loaded encoders
        for col in feature_columns: # Iterate through expected feature columns
            if col in encoders and col != 'income': # Check if it's a categorical column that needs encoding
                if col in processed_batch_data.columns: # Ensure column exists in batch data
                    # Handle unseen labels: map to a default or use a more robust strategy
                    # Check if the value is in the encoder's classes before transforming
                    processed_batch_data[col] = processed_batch_data[col].apply(
                        lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else np.nan # Use np.nan for unseen
                    )
                else:
                     st.warning(f"Categorical feature '{col}' expected but not found in batch data.")
                     processed_batch_data[col] = np.nan # Add missing categorical column as NaN


        # Ensure column order matches the training data
        processed_batch_data = processed_batch_data[feature_columns]

        # Fill any remaining NaNs that might have been introduced during processing (e.g., from unseen categories)
        # Using 0 here, but a more sophisticated approach might use mean/median from training data
        processed_batch_data.fillna(0, inplace=True)


        # Apply scaling to the processed batch data using the loaded scaler
        try:
            # Identify numerical columns among the selected features for scaling
            numerical_cols_to_scale = ['age', 'hours-per-week', 'capital-gain', 'capital-loss'] # Ensure this matches the training features

            # Ensure these columns exist in the batch data before attempting to scale
            cols_to_scale_present = [col for col in numerical_cols_to_scale if col in processed_batch_data.columns]
            if cols_to_scale_present:
                batch_scaled_numerical = scaler.transform(processed_batch_data[cols_to_scale_present])
                # Replace original numerical columns with scaled ones
                processed_batch_data[cols_to_scale_present] = batch_scaled_numerical
            else:
                 st.warning("No numerical columns found for scaling in batch data.")


            batch_scaled_df = processed_batch_data.copy() # Renaming for clarity

        except Exception as e:
            st.error(f"Error during scaling batch input data: {e}")
            st.stop()


        # Make batch predictions
        batch_preds_encoded = model.predict(batch_scaled_df)

        # Decode batch predictions using the income encoder
        if 'income' in encoders:
            batch_preds = encoders['income'].inverse_transform(batch_preds_encoded)
            # Add predictions to the processed_batch_data DataFrame
            processed_batch_data['PredictedClass'] = batch_preds
        else:
            st.warning("Income encoder for 'income' not found. Displaying raw predictions.")
            processed_batch_data['PredictedClass'] = batch_preds_encoded


        st.write("✅ Predictions:")
        # Display the processed batch data with predictions
        st.write(processed_batch_data.head())

        # Provide download link for the predicted CSV
        # Use processed_batch_data which now includes the predictions
        csv = processed_batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

    except Exception as e:
        st.error(f"Error during batch prediction process: {e}")
