import redis
import numpy as np
import struct

class RedisHandler():
    def __init__(self, host="localhost", port=6379, decode_responses=True):
        self.r = redis.Redis(host=host, port=port, decode_responses=decode_responses)
    
    def store_array(self, key, array):
        h,w = array.shape
        shape = struct.pack('>II',h,w)
        encoded = shape + array.tobytes()
        self.r.set(key, encoded)

    def get_array(self, key):
        encoded = self.r.get(key)
        h,w = struct.unpack('>II',encoded[:8])
        array = np.frombuffer(encoded, dtype=np.uint8, offset=8).reshape(h,w)
        return array
