from src.predictor import predict_email_multi

print(" Smart Email Classifier (Multi-task: Spam + Priority)")
print("Type 'exit' to quit.\n")

model_type = "svm"  # or "nb" if you used Naive Bayes

while True:
    email = input("Enter an email message: ").strip()
    if email.lower() == "exit":
        break
    if not email:
        print(" Please enter a valid message.\n")
        continue

    results = predict_email_multi(email, model_type=model_type)

    print("\n Prediction Results:")
    for task, output in results.items():
        print(f"  {task.upper()}: {output['label']} (Confidence: {output['confidence']:.2f})")

    print("\n" + "-" * 50 + "\n")