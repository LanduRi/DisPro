import argparse
import sys
sys.path.append('../app/src/main')
from Train import train

def test_train(dataset_root_dir):
    train(dataset_root_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAD')
    parser.add_argument('--dataset_root_dir', type=str, default='./', help='root dir of dataset')
    args = parser.parse_args()
    test_train(args.dataset_root_dir)