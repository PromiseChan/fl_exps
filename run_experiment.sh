set -x
declare -a PRUNING_RATE=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.90 0.95 0.99)
declare -a CL_EPOCH=(1 10)
declare -a CHAIN_EPOCH=(20 30 40 50 60 70 80 90 100)
declare -a ALPHA=(1)

CL_BS=16
FMAPS=64
CIFAR_10_CL_LR=0.01
NUM_ROUNDS=100
SAMP_CLIENT=0.4
PYTHON_CMD="python main.py --num_rounds $NUM_ROUNDS --cl_bs $CL_BS --feature_maps $FMAPS --samp_rate $SAMP_CLIENT --alpha $ALPHA ";


DATASET="cifar10"

#Cifar10
for a in "${ALPHA[@]}";
do
  for ce in "${CHAIN_EPOCH[@]}";
    do
        echo "client chain epochs: $cle"
        ##Baseline
        $PYTHON_CMD --cl_lr $CIFAR_10_CL_LR --model resnet12 --wbits 32 --cl_epochs 1 --dataset $DATASET --alpha $a --val_ratio 0.1 --chain_epochs $ce

#        ##Binary
#        $PYTHON_CMD --cl_lr $CIFAR_10_CL_LR --model resnet8 --wbits 1 --bnn --cl_epochs $cle --dataset $DATASET --alpha $a
#
#            ##Quant
#        $PYTHON_CMD --cl_lr $CIFAR_10_CL_LR --model qresnet8 --wbits 4 --cl_epochs $cle --dataset $DATASET --alpha $a
#        $PYTHON_CMD --cl_lr $CIFAR_10_CL_LR --model qresnet8 --wbits 8 --cl_epochs $cle --dataset $DATASET --alpha $a

#        for p in "${PRUNING_RATE[@]}"
#        do
#            $PYTHON_CMD --cl_lr $CIFAR_10_CL_LR --model resnet8 --cl_epochs $cle --dataset $DATASET --prune --prate $p --prune_srv --alpha $a
#        done
    done
done

#Cifar100

#CIFAR_100_CL_LR=0.03
#
#for a in "${ALPHA[@]}"
#    for cle in "${CL_EPOCH[@]}"
#    do
#
#        ##Baseline
#        $PYTHON_CMD --cl_lr $CIFAR_100_CL_LR --model resnet8 --wbits 32 --cl_epochs $cle --dataset $DATASET --alpha $a
#
#        ##Binary
#        $PYTHON_CMD --cl_lr $CIFAR_100_CL_LR --model resnet8 --wbits 1 --bnn --cl_epochs $cle --dataset $DATASET --alpha $a
#
#            ##Quant
#        $PYTHON_CMD --cl_lr $CIFAR_100_CL_LR --model qresnet8 --wbits 4 --cl_epochs $cle --dataset $DATASET --alpha $a
#        $PYTHON_CMD --cl_lr $CIFAR_100_CL_LR --model qresnet8 --wbits 8 --cl_epochs $cle --dataset $DATASET --alpha $a
#
#        for p in "${PRUNING_RATE[@]}"
#        do
#            $PYTHON_CMD --cl_lr $CIFAR_100_CL_LR --model resnet8 --cl_epochs $cle --dataset $DATASET --prune --prate $p --prune_srv --alpha $a
#        done
#    done
#done