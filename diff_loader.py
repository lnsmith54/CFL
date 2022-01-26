diff --git a/timm/data/loader.py b/timm/data/loader.py
index a02399a..ce5247b 100644
--- a/timm/data/loader.py
+++ b/timm/data/loader.py
@@ -9,8 +9,10 @@ import random
 from functools import partial
 from typing import Callable
 
+import torch
 import torch.utils.data
 import numpy as np
+import csv
 
 from .transforms_factory import create_transform
 from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
@@ -148,6 +150,8 @@ def create_loader(
         dataset,
         input_size,
         batch_size,
+        train_size=0,
+        read_sampler=None,
         is_training=False,
         use_prefetcher=True,
         no_aug=False,
@@ -205,6 +209,21 @@ def create_loader(
     )
 
     sampler = None
+    if is_training:
+        if train_size>0:
+            sampler = torch.utils.data.SubsetRandomSampler(torch.arange(train_size))
+            print("Training sampler SubsetRandomSampler with ",train_size, " samples.")
+        elif read_sampler is not None:
+            f = open(read_sampler)
+            sam = csv.reader(f)
+            Samples = []
+            for indx in sam:
+                for i in indx:
+                    Samples.append(int(i))
+            samples =  torch.Tensor(Samples).short()
+            sampler = torch.utils.data.SubsetRandomSampler(samples)
+            print("Training sampler SubsetRandomSampler from ",read_sampler)
+
     if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
         if is_training:
             if num_aug_repeats:
