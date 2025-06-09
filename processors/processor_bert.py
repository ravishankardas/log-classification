from sentence_transformers import SentenceTransformer, util
import joblib

transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier_model = joblib.load('models/log_classifier.joblib')


def classify_with_bert(log_message):

    # Encode the input log message into a vector embedding
    message_embedding = transformer_model.encode(log_message, convert_to_tensor=True)

    probabilities = classifier_model.predict_proba([message_embedding])[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    predicted_label = classifier_model.classes_[probabilities.argmax()]
    return predicted_label



if __name__ == "__main__":
    # Example usage
    log_messages = [
        "User User123 logged in.",
        "Backup started at 2023-10-01 12:00:00.",
        "Backup completed successfully.",
        "System updated to version 1.2.3.",
        "File report.pdf uploaded successfully by user User456.",
        "Disk cleanup completed successfully.",
        "System reboot initiated by user User789.",
        "Account with ID 1001 created by admin.",
        "Hey brothear, how are you?",
    ]
    
    for message in log_messages:
        print(f"{message} => {classify_with_bert(message)}")