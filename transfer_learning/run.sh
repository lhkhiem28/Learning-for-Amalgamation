
## tiki
python run.py --pretrained_model="bert" --head_model="fc" --dataset="tiki" --num_epochs=30 --batch_size=16 --learning_rate=1e-5 --nrows=20000
python run.py --pretrained_model="bert" --head_model="lstm-attn" --dataset="tiki" --num_epochs=30 --batch_size=16 --learning_rate=1e-5 --nrows=20000
python run.py --pretrained_model="bert" --head_model="lstm-cnn" --dataset="tiki" --num_epochs=30 --batch_size=16 --learning_rate=1e-5 --nrows=20000

python run.py --pretrained_model="phobert" --head_model="fc" --dataset="tiki" --num_epochs=30 --batch_size=16 --learning_rate=1e-5 --nrows=20000
python run.py --pretrained_model="phobert" --head_model="lstm-attn" --dataset="tiki" --num_epochs=30 --batch_size=16 --learning_rate=1e-5 --nrows=20000
python run.py --pretrained_model="phobert" --head_model="lstm-cnn" --dataset="tiki" --num_epochs=30 --batch_size=16 --learning_rate=1e-5 --nrows=20000
