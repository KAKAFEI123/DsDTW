# #!/usr/bin/env sh

# Evaluate DeepSignDB, Stylus Scenario
## use "--rf" for random forgery scenario (the same to below)
for seed in $(seq 111 111 555)
do
    echo $seed 
    python evaluate_DeepSignDB_stylus.py --epoch End --seed $seed --train-shot-g 4 #--rf
done
## Using 4 templates
python verify_stylus_all.py --train-shot-g 4 --epoch End
## Using 1 template
python verify_stylus_all.py --train-shot-g 1 --epoch End


## Evaluate DeepSignDB, Finger Scenario
for seed in $(seq 111 111 555)
do
    echo $seed 
    python evaluate_DeepSignDB_finger.py --epoch End --seed $seed --train-shot-g 4 #--rf
done
## Using 4 templates
python verify_finger_all.py --train-shot-g 4 --epoch End
## Using 1 template
python verify_finger_all.py --train-shot-g 1 --epoch End

