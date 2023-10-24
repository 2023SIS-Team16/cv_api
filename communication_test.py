import communication

communicator = communication.Communication()

sample_string = "this is a test sentnce for the natural language processing model"

for letter in sample_string:
    communicator.new_letter(letter)
