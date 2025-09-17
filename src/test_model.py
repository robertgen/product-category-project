import joblib
import pandas as pd

# Load the saved model
model = joblib.load("product-category-project/model/sentiment_model.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    # Create a DataFrame from input (doar product_title)
    user_input = pd.DataFrame([{"product_title": title}])

    # Predict category
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)