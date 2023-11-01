import requests
import json
# import main

#For communicating with NLP over HTTP requests
request_url = "http://127.0.0.1:5000/parse_text"

#To be called once the model has found a new letter
class Communication:
    string = ""

    def new_letter_no_conversion(self, letters):
        self.string = self.string + ''.join(letters)

    def new_letter(self, letter):
        print(f"New Letter: {letter}")

        self.string = self.string + letter

        input = {
            "text": self.string,
            "index": 0,
            "truncate": False,
        }
        json_input = json.dumps(input)
        response = requests.get(request_url, json=json_input)
        # print(response)
        # print(response.content)
        content = json.loads(response.content)
        # print(content)

        #Truncate string if there's a response
        # self.string = self.string[content['index']:]
        print(f"String: {self.string}")

        print(content.keys())

        if "text" in content:
            print("Message emitted")
            print(content['text'])
            #main.socketio.emit_message(content['text'])
            return(content['text'])
        return None

    def reset_string(self):
        self.string = ""

    def upload_to_interface(self, text_to_display):
        print("Uploading text to the HoloLens")