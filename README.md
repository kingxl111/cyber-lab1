python3 train_baseline.py
python3 train_improved.py
python3 train_torchvision.py --model resnet18 --variant baseline
python3 train_torchvision.py --model vit_b_16 --variant baseline
python3 train_torchvision.py --model resnet18 --variant improved
python3 train_torchvision.py --model vit_b_16 --variant improved
python3 train_custom_model.py
python3 compare_results.py

```bash
python3 train_torchvision.py --model resnet18 --variant baseline
Device: cuda
Epoch 01 | train_loss=0.6372 | val_loss=0.5896
Epoch 02 | train_loss=0.5914 | val_loss=0.5973
Epoch 03 | train_loss=0.5815 | val_loss=0.5723
Epoch 04 | train_loss=0.5762 | val_loss=0.5716
Epoch 05 | train_loss=0.5677 | val_loss=0.5770
Epoch 06 | train_loss=0.5664 | val_loss=0.5782
Epoch 07 | train_loss=0.5631 | val_loss=0.5800
Early stopping. Best epoch: 4

Best threshold on validation: 0.59
Validation metrics at best threshold:
accuracy: 0.7448
precision: 0.3176
recall: 0.6154
f1: 0.4189
roc_auc: 0.7703

TEST METRICS (resnet18_baseline)
accuracy: 0.7482
precision: 0.3234
recall: 0.6257
f1: 0.4264
roc_auc: 0.7746
```



```bash
python3 train_torchvision.py --model vit_b_16 --variant baseline
Device: cuda
Epoch 01 | train_loss=0.7336 | val_loss=0.6377
Epoch 02 | train_loss=0.6317 | val_loss=0.6404
Epoch 03 | train_loss=0.6243 | val_loss=0.6327
Epoch 04 | train_loss=0.6203 | val_loss=0.6099
Epoch 05 | train_loss=0.6082 | val_loss=0.6092
Epoch 06 | train_loss=0.6028 | val_loss=0.5965

Best threshold on validation: 0.57
Validation metrics at best threshold:
accuracy: 0.7391
precision: 0.3062
recall: 0.5886
f1: 0.4028
roc_auc: 0.7449

TEST METRICS (vit_b_16_baseline)
accuracy: 0.7435
precision: 0.3151
recall: 0.6090
f1: 0.4153
roc_auc: 0.7542

```


```bash
python3 train_torchvision.py --model resnet18 --variant improved
Device: cuda
Epoch 01 | train_loss=0.6264 | val_loss=0.5856
Epoch 02 | train_loss=0.5892 | val_loss=0.5802
Epoch 03 | train_loss=0.5720 | val_loss=0.5678
Epoch 04 | train_loss=0.5633 | val_loss=0.5649
Epoch 05 | train_loss=0.5510 | val_loss=0.5648
Epoch 06 | train_loss=0.5472 | val_loss=0.5596
Epoch 07 | train_loss=0.5419 | val_loss=0.5617
Epoch 08 | train_loss=0.5315 | val_loss=0.5654

Best threshold on validation: 0.60
Validation metrics at best threshold:
accuracy: 0.7587
precision: 0.3371
recall: 0.6355
f1: 0.4405
roc_auc: 0.7842

TEST METRICS (resnet18_improved)
accuracy: 0.7554
precision: 0.3344
recall: 0.6410
f1: 0.4395
roc_auc: 0.7843

```


```bash
time python3 train_torchvision.py --model vit_b_16 --variant improved
Device: cuda
Epoch 01 | train_loss=0.6972 | val_loss=0.6205
Epoch 02 | train_loss=0.6227 | val_loss=0.6141
Epoch 03 | train_loss=0.6407 | val_loss=0.6427
Epoch 04 | train_loss=0.6052 | val_loss=0.5929
Epoch 05 | train_loss=0.5970 | val_loss=0.6063
Epoch 06 | train_loss=0.5961 | val_loss=0.5915

Best threshold on validation: 0.57
Validation metrics at best threshold:
accuracy: 0.7241
precision: 0.3027
recall: 0.6488
f1: 0.4129
roc_auc: 0.7531

TEST METRICS (vit_b_16_improved)
accuracy: 0.7253
precision: 0.3066
recall: 0.6631
f1: 0.4194
roc_auc: 0.7599

real	13m1,640s
user	13m53,423s
sys	0m14,745s

```