import re
import random



class RuleBasedChatbot:
    def __init__(self):
        # Define patterns and responses
        self.rules = [
            {
                'patterns': [r'hi|hello|hey', r'howdy|greetings'],
                'responses': ['Hello!', 'Hi there!', 'Greetings!'],
                'context': 'greeting'
            },
            {
                'patterns': [r'bye|goodbye|see ya'],
                'responses': ['Goodbye!', 'See you later!', 'Have a nice day!'],
                'context': 'farewell'
            },
            {
                'patterns': [r'how are you|how\'s it going'],
                'responses': ['I\'m just a chatbot, but I\'m functioning well!', 'All systems operational!'],
                'context': 'status'
            },
            {
                'patterns': [r'what\'s your name|who are you'],
                'responses': ['I\'m a simple rule-based chatbot.', 'You can call me ChatBot.'],
                'context': 'identity'
            },
            {
                'patterns': [r'thanks|thank you'],
                'responses': ['You\'re welcome!', 'Happy to help!'],
                'context': 'gratitude'
            },
            {
                'patterns': [r'(.*) weather (.*)'],
                'responses': ['I don\'t have real-time weather data, sorry!', 'You might want to check a weather website for that.'],
                'context': 'weather'
            },
            {
                'patterns': [r'(.*) time (.*)'],
                'responses': ['I don\'t have access to the current time.', 'My internal clock isn\'t working, sorry!'],
                'context': 'time'
            }
        ]
        
        # Default response if no pattern matches
        self.default_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "I don't have a response for that.",
            "Interesting. Tell me more about something else."
        ]
        
        # Track context for simple conversation flow
        self.current_context = None
    
    def respond(self, user_input):
        user_input = user_input.lower().strip()
        
        # Check for empty input
        if not user_input:
            return "You didn't say anything!"
        
        # Check each rule for matching patterns
        for rule in self.rules:
            for pattern in rule['patterns']:
                if re.search(pattern, user_input):
                    self.current_context = rule['context']
                    return random.choice(rule['responses'])
        
        # If no pattern matches, return a default response
        return random.choice(self.default_responses)
    
    def chat(self):
        print("ChatBot: Hello! I'm a simple rule-based chatbot. Type 'quit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("ChatBot: Goodbye!")
                break
            response = self.respond(user_input)
            print(f"ChatBot: {response}")

# Create and run the chatbot
if __name__ == "__main__":
    bot = RuleBasedChatbot()
    bot.chat()