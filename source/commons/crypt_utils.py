import os
import sys
import hashlib
import base64

from Crypto import Random
from Crypto.Cipher import AES

def read_binary(filename: str) -> tuple[bytes, bool]:
    """ 
    Read the binary file
    
    Args:
        filename: name of the file to read
    Returns:
        read file bytes
        bool whether reading was successful
    """

    with open(filename, "rb") as f:
        key = f.read()
        keyf = True
        return key, keyf
    

class AESCipher:

    def __init__(self, 
                 iv_file=os.getenv("IV_FILE"),
                 key_file=os.getenv("KEY_FILE")):
        """ 
        Args:
            iv_file: files for decrypting the postgreSQL db password
            key_file: files for decrypting the postgreSQL db password
        """

        self.bs = AES.block_size
        self.iv_file = iv_file
        self.key_file = key_file

        try:
            self.iv, self.ivf = read_binary(iv_file)
        except:
            with open(iv_file, "wb") as f:
                self.iv = Random.new().read(AES.block_size)
                f.write(self.iv)
            self.ivf = False

        try:
            self.key, self.keyf = read_binary(key_file)
        except:
            with open(key_file, "wb") as f:
                self.key = os.urandom(32)
                f.write(self.key)
            self.keyf = False
        self.key = hashlib.sha256(self.key).digest()

    
    def encrypt(self, raw: str) -> str:
        """ Encrypt the input string """

        raw = self._pad(raw)
        iv = self.iv
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw.encode())).decode()
    

    def decrypt(self, enc: str) -> str:
        """ Decrypt the input string """

        enc = base64.b64decode(enc)
        iv = enc[:self.bs]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[self.bs:])).decode("utf-8")
    

    def _pad(self, s: str):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)
    

    def clean(self):
        """ Delete key and ivf files """
        if not self.keyf:
            os.remove(self.key_file)
        if not self.ivf:
            os.remove(self.iv_file)

    @staticmethod
    def _unpad(s: str):
        return s[:-ord(s[len(s) - 1:])]
    

if __name__ == "__main__":
    string = sys.argv[2]
    c = AESCipher()
    if sys.argv[1] == "e":
        e = c.encrypt(string)
    else:
        e = c.decrypt(string)
    print(e)