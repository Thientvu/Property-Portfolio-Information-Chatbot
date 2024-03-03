import json
from datetime import datetime

class ChatHistory:
    def __init__(self, filename="chat_history.json"):
        self.filename = filename

    def log_message(self, message, sender):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": timestamp, "sender": sender, "message": message}
        with open(self.filename, "a") as file:
            file.write(json.dumps(log_entry) + "\n")

    def retrieve_history(self):
        with open(self.filename, "r") as file:
            history = [json.loads(line) for line in file]
        return history

# Example usage
chat_history = ChatHistory()
chat_history.log_message("Hello, how can I help you?", "chatbot")
chat_history.log_message("I need some assistance.", "user")
print(chat_history.retrieve_history())
