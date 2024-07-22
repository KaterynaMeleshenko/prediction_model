import pandas as pd
import joblib  

# Load the trained model
model = joblib.load('model.pkl') 

# Load the test data
test_data = pd.read_csv('data/hidden_test.csv')

# Apply polynomial feature
test_data['6'] = test_data['6'] ** 2

# Make predictions
predictions = model.predict(test_data[['6']])

# Create a DataFrame to hold the results
results = pd.DataFrame({
    'Id': test_data.index,  
    'Prediction': predictions
})

# Save predictions to a CSV file
results.to_csv('predictions.csv', index=False)