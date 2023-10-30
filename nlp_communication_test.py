import nlp_communication
import unittest

communicator = nlp_communication.Communication()

sample_string = "this is a test sentnce for the natural language processing model"

for letter in sample_string:
    communicator.new_letter(letter)

class TestCommunication(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.communicator = nlp_communication.Communication()
        print("Creating Class")

    def check_string(self, string):
        self.communicator.reset_string()
        for letter in string:
            response = communicator.new_letter(letter)
            print(response)

    def test_string_1(self):
        self.check_string("this is a test")
        
    def test_string_2(self):
        self.check_string("how are you going")

    def test_string_3(self):
        self.check_string("this is quite a long sentence indeed and repeats itself too much such as by talking about how long it is more than once which is why it is far too long")

    def test_string_4(self):
        self.check_string("the car is slow")

    def test_string_5(self):
        self.check_string("ths sntexce is hrd to prse")

    def test_string_6(self):
        self.check_string("messages should be sent over websockets using socketio")




if __name__ == '__main__':
    unittest.main()