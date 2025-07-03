from src.predictor import predict_email_multi

# Sample test emails
test_emails = [
    "Congratulations! You have won a free iPhone. Click the link to claim it now!",
    "VPN issue reported to IT, needs urgent attention.",
    "Team meeting postponed to 3PM. Not urgent.",
    "Final reminder to update your billing info to avoid service disruption.",
    "This is a newsletter from your subscribed tech blog.",
]

# Model type can be 'svm' or 'nb'
model_type = "svm"

for idx, email in enumerate(test_emails):
    print(f"\n--- Email {idx + 1} ---")
    print(f"Text: {email}")
    results = predict_email_multi(email, model_type=model_type)

    print("Prediction Results:")
    for task, output in results.items():
        print(f"  {task.upper()}: {output['label']} (Confidence: {output['confidence']:.2f})")
