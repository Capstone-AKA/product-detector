from dotenv import load_dotenv
import os

def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y", "t") 

load_dotenv()

PD = os.getenv('PRODUCT_DETECTOR')
HOST_URL = os.getenv('HOST_URL')
DEBUG = str2bool(os.getenv('DEBUG', '0'))