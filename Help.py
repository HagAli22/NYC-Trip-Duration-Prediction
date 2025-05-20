import os
import json
import joblib


def update_baseline_metadata(model_data):
    dataset_metadata_path = f"Process_data/data_overview.json"
    dataset_metadata = {}
    with open(dataset_metadata_path, 'r') as f:
        dataset_metadata = json.load(f)
    baseline_metadata_path = "Best_model_data_overview.json"
    if os.path.exists("Best_model_data_overview.json"):
        with open("Best_model_data_overview.json", 'r') as f:
            baseline_metadata = json.load(f)
    else:
        baseline_metadata = {}

    if "Best_model" not in baseline_metadata:
        baseline_metadata["Best_model"] = {}

   
    if "Best_model" not in baseline_metadata or "test_r2" not in baseline_metadata["Best_model"]:
        # First time we save the model without comparison
        baseline_metadata["Best_model"] = {
            "train_rmse": model_data["train_rmse"],
            "train_r2": model_data["train_r2"],
            "test_r2": model_data["test_r2"],
            "test_rmse": model_data["test_rmse"],
            "selected_feature_names": model_data["selected_feature_names"],
            "dataset_metadata": dataset_metadata
        }
        # Manage history if current model is not the best
        history = {}
        if os.path.exists("history.json"):
            with open("history.json", 'r') as f:
                history = json.load(f)
        
        id = history.get("counter", 0)
        history["counter"] = id + 1
        history[id] = {
            'train_rmse': model_data["train_rmse"],
            'train_r2':   model_data["train_r2"],
            'test_rmse':  model_data["test_rmse"],
            'test_r2':    model_data["test_r2"],
            'selected_feature_names': model_data["selected_feature_names"],
            'dataset_metadata': dataset_metadata
        }

        # Save updated history
        with open("history.json", 'w') as f:
            json.dump(history, f, indent=4)


    elif baseline_metadata["Best_model"]["test_r2"] < model_data["test_r2"]:
        # We save the model if it is better
        baseline_metadata["Best_model"] = {
            "train_rmse": model_data["train_rmse"],
            "train_r2": model_data["train_r2"],
            "test_r2": model_data["test_r2"],
            "test_rmse": model_data["test_rmse"],
            "selected_feature_names": model_data["selected_feature_names"],
            "dataset_metadata": dataset_metadata
        }

    # In both cases, save the file.
    with open(baseline_metadata_path, 'w') as f:
        json.dump(baseline_metadata, f, indent=4)

    
    # Manage history if current model is not the best
        history = {}
        if os.path.exists("history.json"):
            with open("history.json", 'r') as f:
                history = json.load(f)
        
        id = history.get("counter", 0)
        history["counter"] = id + 1
        history[id] = {
            'train_rmse': model_data["train_rmse"],
            'train_r2':   model_data["train_r2"],
            'test_rmse':  model_data["test_rmse"],
            'test_r2':    model_data["test_r2"],
            'selected_feature_names': model_data["selected_feature_names"],
            'dataset_metadata': dataset_metadata
        }

        # Save updated history
        with open("history.json", 'w') as f:
            json.dump(history, f, indent=4)

    
        



def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        print("Error: The file could not be loaded.")
    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    ...


  
    
    
   