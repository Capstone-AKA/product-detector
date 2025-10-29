from dotenv import load_dotenv
import os

def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t") 

load_dotenv()

PD = os.getenv('PRODUCT_DETECTOR')
CART_ID = os.getenv('CART_ID')
HOST_URL = os.getenv('HOST_URL')
ADD_ENDPOINT = os.getenv('ADD_ENDPOINT')
DEBUG = str2bool(os.getenv('DEBUG', '0'))