# bias-mitigation-sgd
Applying Fairlearn's bias mitigation techniques to a Neural Net

Example of run command:
`python main.py --device cpu --dataset=adult --sensitive=education -rn=adult-DP_SGD-edu`

Running dp_sgd on noise: 1.0 with adult dataset

python main.py --device cpu --dataset='adult' --sensitive='education' -rn=adult-DP_SGD-edu
