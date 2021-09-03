import argparse
import sys
sys.path.append('../app/src/main')
from eval import evaluate

def test_eval(dataset_root_dir):
    eval_result = evaluate(dataset_root_dir)
    print('AUC: ', eval_result*100, '%')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VAD')
    parser.add_argument('--dataset_root_dir', type=str, default='./', help='root dir of dataset')
    args = parser.parse_args()
    test_eval(args.dataset_root_dir)