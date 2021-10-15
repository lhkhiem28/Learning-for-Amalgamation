#export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=1


# tiki
#nohup python -u run.py --pretrained_model="bert"     --dataset="tiki" --batch_size=8 --nrows=5000 >> tiki_5K_bert.out &
#nohup python -u run.py --pretrained_model="phobert"  --dataset="tiki" --batch_size=8  --nrows=5000 >> tiki_5K_phobert.out &
#nohup python -u run.py --pretrained_model="lstmcnn"  --dataset="tiki" --batch_size=8  --nrows=5000 --num_epochs=200 --learning_rate=1e-3 >> tiki_5K_lstmcnn.out &
#nohup python -u run.py --pretrained_model="moe"      --dataset="tiki" --batch_size=32  --nrows=25000 --model_name='Moe_Concate' >> tiki_5K_moe_concate.out &
#nohup python -u run.py --pretrained_model="moe"      --dataset="tiki" --batch_size=32  --nrows=5000 --learning_rate=1e-4 --model_name='Moe_Gating' >> tiki_5K_moe_gating.out &

# aivivn
#python -u run.py --pretrained_model="bert"      --dataset="aivivn" --batch_size=8
#python -u run.py --pretrained_model="phobert"   --dataset="aivivn" --batch_size=8
#python -u run.py --pretrained_model="lstmcnn"   --dataset="aivivn" --batch_size=8
#nohup python -u run.py --pretrained_model="moe"       --dataset="aivivn" --batch_size=8   --model_name='Moe_Gating' >> tiki_moe_gating.out &

python -u run.py --pretrained_model="lstmcnn" --dataset="books" --batch_size=8 --learning_rate=1e-3
python -u run.py --pretrained_model="lstmcnn" --dataset="dvd" --batch_size=8 --learning_rate=1e-3
python -u run.py --pretrained_model="lstmcnn" --dataset="electronics" --batch_size=8 --learning_rate=1e-3
python -u run.py --pretrained_model="lstmcnn" --dataset="kitchen_&_housewares" --batch_size=8 --learning_rate=1e-3