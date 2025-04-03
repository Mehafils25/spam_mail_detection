import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load the trained model
model_path = 'C:\\Users\\mehuk\\OneDrive\\Desktop\\project sample\\logistic_regression_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict():
    try:
        inputs = [float(entry.get()) for entry in input_fields]
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)
        result_label.config(text=f"Predicted Output: {prediction[0]:.4f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Create GUI window
root = tk.Tk()
root.title("Linear Regression Model GUI")
root.geometry("500x400")
root.configure(bg="#1E1E2E")  # Vibrant background

# Frame for input fields
input_frame = tk.Frame(root, bg="#282A36", bd=5, relief=tk.RIDGE)
input_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

num_features = 1  # Change this based on your model input size
input_fields = []

for i in range(num_features):
    label = tk.Label(input_frame, text=f"Feature {i+1}:", font=("Arial", 12, "bold"), fg="#F8F8F2", bg="#282A36")
    label.pack(pady=5)
    entry = tk.Entry(input_frame, font=("Arial", 12), bg="#44475A", fg="white", relief=tk.FLAT)
    entry.pack(pady=5, ipadx=10, ipady=5)
    input_fields.append(entry)

# Styled Predict Button
predict_button = tk.Button(root, text="Predict", command=predict, font=("Arial", 14, "bold"), bg="#50FA7B", fg="#282A36", relief=tk.GROOVE, bd=3, padx=10, pady=5)
predict_button.pack(pady=10)

# Label for result
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="#FF79C6", bg="#1E1E2E")
result_label.pack(pady=10)

# Run the application
root.mainloop()
