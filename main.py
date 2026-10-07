"""
Synthetic Fashion DCGAN - Main Entry Point & Command Dispatcher (Stage 7)
"""

import sys
import utils.config as config
from dashboard import run_dashboard
from evaluation.evaluate import run_evaluation
from training import train_dcgan


def main():
    if len(sys.argv) > 1:
        flag = sys.argv[1].lower()
        if flag == "--full":
            print("Running full 25-epoch DCGAN training...")
            train_dcgan(num_epochs=config.NUM_EPOCHS, is_sanity_test=False)
        elif flag == "--dashboard":
            print("Launching interactive Flask dashboard server...")
            run_dashboard()
        elif flag == "--evaluate":
            run_evaluation()
        else:
            print(f"Unknown option '{sys.argv[1]}'. Running evaluation by default...")
            run_evaluation()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
