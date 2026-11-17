# Explaining the solution

**Why use convolutions?** A small kernel learns local patterns and shares its weights across spatial positions. Deeper layers combine edges and textures into larger patterns. Pooling reduces spatial size and computation, while losing some precise location information.

**What is the baseline's tensor shape?** For a batch of B images, input is B x 3 x 150 x 150. The three pooled feature maps are B x 16 x 75 x 75, B x 32 x 37 x 37, and B x 64 x 18 x 18. Flattening gives 20,736 features. The final output is B raw logits.

**Why no sigmoid in the model?** BCEWithLogitsLoss combines sigmoid and binary cross-entropy in a numerically stable operation. During prediction, sigmoid converts a logit to the model's estimated dog probability. A probability threshold of 0.5 equals a logit threshold of zero.

**Why can validation accuracy exceed training accuracy?** Dropout is active during training but disabled during evaluation. The contest model also sees harder, randomly augmented training images. Training metrics are collected while weights change through the epoch, whereas validation uses the final epoch weights.

**Why select minimum validation loss rather than the final epoch?** Training accuracy can improve while generalization worsens. BCE also penalizes confident errors, so the epoch with best BCE can differ from the one with best accuracy. The selection rule was fixed before examining test scores.

**What exactly is frozen in transfer learning?** All backbone parameters have requires_grad=False. The feature extractor remains in eval mode so ResNet's BatchNorm running statistics do not change. Only the newly added classification head is optimized. The original ImageNet 1,000-class classifier is discarded.

**Why cache features?** Frozen weights plus deterministic preprocessing produce the same features on each epoch. Computing them once avoids repeated expensive convolutions. Caches are stored in float16; the resulting small numerical approximation is documented. Randomly augmented images would require recomputing features or explicitly caching augmentation variants.

**Why is the contest model eligible?** Its convolutions, normalization layers and head are initialized randomly. It uses only the cleaned course training split, fixed normalization constants and training augmentation. It never loads ImageNet weights, cached pretrained features, teacher outputs or a pretrained ensemble member.

**Why remove training duplicates?** Exact matching photographs in training and evaluation can inflate performance through memorization. The audit found five redundant training copies, including three groups crossing split boundaries. Removing training copies preserves the instructor's evaluation set.

**What do the intervals and custom photo establish?** Wilson intervals summarize uncertainty in the test accuracy, with a caveat for one duplicated test photograph. They do not capture variation between random training seeds or prove that two models differ significantly. The custom dog photograph demonstrates the complete inference pipeline; one photograph is not an accuracy benchmark.

**What would improve the contest result next?** Use validation data to compare higher-resolution training, additional random seeds, or an ensemble containing only models trained from scratch. Measure the improvement before making a claim, and do not repeatedly tune against the test set.
