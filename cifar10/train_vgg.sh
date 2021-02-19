#python main.py --arch vgg_16_bn \
#--use_pretrain  \
#--bit 5 --gpu 1 --select_method learned --keep_freq 0.85 --epoch 300

#python main.py --arch vgg_7_bn \
#--bit 32 --gpu 1 --select_method learned --keep_freq 0.85 --epoch 350

python main.py --arch vgg_7_bn \
--use_pretrain  --batch_size 128 \
--bit 5 --gpu 1

python main.py --arch vgg_7_bn \
--use_pretrain  --batch_size 128 \
--bit 4 --gpu 1

python main.py --arch vgg_7_bn \
--use_pretrain  --batch_size 128 \
--bit 3 --gpu 1

#python main.py --arch vgg_7_bn \
#--use_pretrain --batch_size 128 \
#--bit 2 --gpu 1