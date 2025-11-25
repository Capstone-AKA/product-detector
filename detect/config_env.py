from dotenv import load_dotenv
import os

def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t") 

load_dotenv()

PD = os.getenv('PRODUCT_DETECTOR')
CART_ID = int(os.getenv('CART_ID'))
HOST_URL = os.getenv('HOST_URL')
ADD_ENDPOINT = os.getenv('ADD_ENDPOINT')
DEBUG = str2bool(os.getenv('DEBUG', '0'))
TIME_STAMP = str2bool(os.getenv('TIME_STAMP', '0'))
CAM_WIDTH = int(os.getenv('CAM_WIDTH'))
CAM_HEIGHT = int(os.getenv('CAM_HEIGHT'))