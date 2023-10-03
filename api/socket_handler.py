import uuid

class SocketHandler:
    transmission_in_progress = False

    def __init__(self, socket, redis):
        self.socket = socket
        self.handler = redis

    def create_transmission(self) -> str:
        # create in redis
        return str(uuid.uuid4())

    def handle_image(self, uuid_str, image):
        if uuid_str == None:
            raise Exception("No transmission in progress")
        current_array = self.handler.get_array(uuid)
        new_array = self.convert_encoded_image(image)
        return uuid

    def verify_transmission(self):
        return True
