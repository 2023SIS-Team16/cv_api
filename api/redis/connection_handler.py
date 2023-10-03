import redis
import uuid
import numpy as np
import struct

class RedisConnection():
    def __init__(self, connection):
        self.connection = connection
    
    def __del__(self):
        self.connection.close()
    
    def check_transmission_exists(self, uuid):
        return self.connection.exists(uuid)

    def create_transmission(self, uuid):
        self.connection.hset('identifier:{uuid}', mapping = {
            'status': 0,
            'data': None,
        })
    
    def get_transmission_block(self, uuid):
        return self.connection.hget('identifier:{uuid}')
    
    def add_frame_to_transmission(self, uuid, frame):
        block = self.connection.hget('identifier:{uuid}')
        if block is None:
            raise Exception("Transmission does not exist")
        
        if 'status' not in block or block['status'] != 0 or block['status'] != 1:
            raise Exception("Transmission is not in a mutable state")

        if 'data' not in block or block['data'] is None:
            array = np.array([frame])
            encoded = self.__convert_to_bytes(array)
        else:
            data_block = block['data']
            array = self.__convert_from_bytes(data_block)
            array = np.append(array, frame, axis=0)
            encoded = self.__convert_to_bytes(array)
        
        block['data'] = encoded
        block['status'] = 1
        self.connection.hset('identifier:{uuid}', block)

    def complete_transmission(self, identifier):
        block = self.connection.hget('identifier:{uuid}')
        if block is None:
            raise Exception("Transmission does not exist")
        
        if 'status' not in block or block['status'] != 1:
            raise Exception("Transmission is not in a completable state")
        
        block['status'] = 2
        self.connection.hset('identifier:{uuid}', block)

    def __convert_to_bytes(self, array):
        h,w = array.shape
        shape = struct.pack('>II',h,w)
        encoded = shape + array.tobytes()
        return encoded
    
    def __convert_from_bytes(self, bytes):
        h,w = struct.unpack('>II', bytes[:8])
        array = np.frombuffer(bytes[8:], dtype=np.uint8).reshape(h,w)
        return array

class RedisConnectionDispatcher():
    connectionPool = None

    def __init__(self, host="localhost", port=6379, db=0):
        self.connectionPool = redis.ConnectionPool(host=host, port=port, db=db)

    def get_connection(self):
        return RedisConnection(connection=redis.Redis(connection_pool=self.connectionPool))