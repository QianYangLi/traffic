import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.dataset import build_or_load_cache
from configs.config import CACHE_PATH
def main():
    build_or_load_cache("data/traffic_martixes",CACHE_PATH)
if __name__=="__main__": main()
