import catboost
import os

# Get the absolute path to the model file
model_path = os.path.abspath(r'c:\Users\LENOVO\Documents\DENICO\Skripsi\Python\Website\Model_Website\catboost_obesity_model.cbm')

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    # Initialize the CatBoostClassifier
    model = catboost.CatBoostClassifier()

    # Load the model
    model.load_model(model_path)

    # Get all model parameters
    params = model.get_all_params()

    # Print the parameters
    print(params)
