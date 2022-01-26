diff --git a/train.py b/train.py
index 10d839b..140779c 100755
--- a/train.py
+++ b/train.py
@@ -33,6 +33,8 @@ from timm.models import create_model, safe_model_name, resume_checkpoint, load_c
     convert_splitbn_model, model_parameters
 from timm.utils import *
 from timm.loss import *
+from timm.loss.asl_focal_loss import *
+from timm.loss.clcarwin_focal_loss import *
 from timm.optim import create_optimizer_v2, optimizer_kwargs
 from timm.scheduler import create_scheduler
 from timm.utils import ApexScaler, NativeScaler
@@ -83,6 +85,10 @@ parser.add_argument('--dataset-download', action='store_true', default=False,
                     help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
 parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                     help='path to class to idx mapping file (default: "")')
+parser.add_argument('--train_size', type=int, default=0, 
+                    help='Number of samples to use for training (default=0 means no limit)')
+parser.add_argument('--read_sampler', type=str, default=None, 
+                    help='CSV File containing training sample numbers (default=None)')
 
 # Model parameters
 parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
@@ -127,10 +133,34 @@ parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                     help='Optimizer momentum (default: 0.9)')
 parser.add_argument('--weight-decay', type=float, default=2e-5,
                     help='weight decay (default: 2e-5)')
+parser.add_argument('--wd_min', default=0, type=float, 
+                    help='Cyclical Weight decay if > 0 (default=0)')
+parser.add_argument('--wd_max', default=1e-3, type=float, 
+                    help='Max WD for cyclical Weight decay (default=1e-3')
+parser.add_argument('--T_min', default=0, type=float, 
+                    help='Cyclical Softmax Temperature if > 0 (default=0, recommended=0.5)')
+parser.add_argument('--T_max', default=2, type=float, 
+                    help='Max Softmax Temperature for cyclical Temperature (default=2')
 parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                     help='Clip gradient norm (default: None, no clipping)')
 parser.add_argument('--clip-mode', type=str, default='norm',
                     help='Gradient clipping mode. One of ("norm", "value", "agc")')
+parser.add_argument('--clip_min', default=0, type=float, 
+                    help='Cyclical gradient clipping if > 0 (default=0, recommended=0.5)')
+parser.add_argument('--clip_max', default=100, type=float, 
+                    help='Max threshold for cyclical gradient clipping (default=100')
+parser.add_argument('--focal_loss', type=str, default='',
+                    help='Focal Loss. One of ("sym", "asym", "cyclical", "asym-cyclical)')
+parser.add_argument('--cyclical_factor', type=float, default=2, 
+                    help='1->Modified focal loss, 2->Cyclical focal loss (default=2)')
+parser.add_argument('--gamma', type=float, default=2, 
+                    help='Symetric focal loss gamma (default=2)')
+parser.add_argument('--gamma0', type=float, default=0, 
+                    help='Cyclical focal loss gamma (default=0)')
+parser.add_argument('--gamma_pos', type=float, default=0, 
+                    help='Asymetric focal loss positive gamma (default=0)')
+parser.add_argument('--gamma_neg', type=float, default=4, 
+                    help='Asymetric focal loss negative gamma (default=4)')
 
 
 # Learning rate schedule parameters
@@ -321,6 +351,7 @@ def _parse_args():
 def main():
     setup_default_logging()
     args, args_text = _parse_args()
+
     
     if args.log_wandb:
         if has_wandb:
@@ -528,6 +559,8 @@ def main():
         dataset_train,
         input_size=data_config['input_size'],
         batch_size=args.batch_size,
+        train_size=args.train_size,
+        read_sampler=args.read_sampler,
         is_training=True,
         use_prefetcher=args.prefetcher,
         no_aug=args.no_aug,
@@ -570,7 +603,17 @@ def main():
     )
 
     # setup loss function
-    if args.jsd_loss:
+    if args.focal_loss=="sym":
+        train_loss_fn = FocalLoss(gamma=args.gamma)
+    elif args.focal_loss=="asym":
+        train_loss_fn = ASLSingleLabel(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg)
+    elif args.focal_loss=="cyclical":
+        train_loss_fn = CFocalLoss(gamma=args.gamma, gamma0=args.gamma0, epochs=num_epochs, 
+                                   factor=args.cyclical_factor)
+    elif args.focal_loss=="asym-cyclical":
+        train_loss_fn = Cyclical_FocalLoss(gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg,
+                        epochs=num_epochs, gamma0=args.gamma0, factor=args.cyclical_factor)
+    elif args.jsd_loss:
         assert num_aug_splits > 1  # JSD only valid with aug splits set
         train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
     elif mixup_active:
@@ -614,6 +657,18 @@ def main():
 
     try:
         for epoch in range(start_epoch, num_epochs):
+            if args.wd_min > 0 or args.clip_min > 0:
+                if args.cyclical_factor*epoch < num_epochs:
+                    eta = 1.0 - args.cyclical_factor *epoch/(num_epochs-1)
+                elif args.cyclical_factor == 1.0:
+                    eta = 0
+                else:
+                    eta = (args.cyclical_factor*epoch/(num_epochs-1) - 1.0) /(args.cyclical_factor - 1.0) 
+            if args.wd_min > 0:
+                optimizer.param_groups[0]['weight_decay'] = (1 - eta)*args.wd_max + eta*args.wd_min
+            elif args.clip_min > 0:
+                args.clip_grad = (1 - eta)*args.clip_max + eta*args.clip_min
+
             if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                 loader_train.sampler.set_epoch(epoch)
 
@@ -673,6 +728,14 @@ def train_one_epoch(
     losses_m = AverageMeter()
 
     model.train()
+    if args.T_min > 0:
+        #   eta = abs(1 - 2*epoch/(args.epochs-1))
+        if args.cyclical_factor*epoch < args.epochs:
+            eta = 1.0 - args.cyclical_factor *epoch/(args.epochs-1)
+        elif args.cyclical_factor == 1.0:
+            eta = 0
+        else:
+            eta = (args.cyclical_factor*epoch/(args.epochs-1) - 1.0) /(args.cyclical_factor - 1.0) 
 
     end = time.time()
     last_idx = len(loader) - 1
@@ -689,7 +752,13 @@ def train_one_epoch(
 
         with amp_autocast():
             output = model(input)
-            loss = loss_fn(output, target)
+            if args.T_min > 0:
+                Temperature = (1 - eta)*args.T_max + eta*args.T_min
+                output = torch.div(output, Temperature)
+            if args.focal_loss.find("cyclical")>-1:
+                loss = loss_fn(output, target, epoch)
+            else:
+                loss = loss_fn(output, target)
 
         if not args.distributed:
             losses_m.update(loss.item(), input.size(0))
