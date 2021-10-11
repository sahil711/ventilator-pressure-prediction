#!/bin/bash
python3 train_classification.py --config ../configs/lstm-classification.yaml --folds  0
python3 train_classification.py --config ../configs/lstm-classification.yaml --folds  1
python3 train_classification.py --config ../configs/lstm-classification.yaml --folds  2
python3 train_classification.py --config ../configs/lstm-classification.yaml --folds  3
python3 train_classification.py --config ../configs/lstm-classification.yaml --folds  4