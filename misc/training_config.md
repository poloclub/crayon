### Waterbirds
<table>
  <tr>
    <td>CRAYON-Attention</td>
    <td>Fine-tune the original model for 10 epochs with a batch size of 128, using the Adam optimizer with a learning rate of 5e-5 and a weight decay of 1e-4. We set the hyperparameters &alpha; and &beta; as 1e7 and 2e5, respectively.</td>
  </tr>
  <tr>
    <td>CRAYON-Pruning</td>
    <td>Prune 1,034 irrelevant neurons in the penultimate layer and trains the last fully connected layer for 10 epochs with a learning rate of 5e-5.</td>
  </tr>
  <tr>
    <td>CRAYON-Attention+Pruning</td>
    <td>Fine-tune the original model for 10 epochs with a batch size of 128, using the Adam optimizer with a learning rate of 5e-5 and a weight decay of 1e-4. We set &alpha; to 1e7 and &beta; to 2e5.</td>
  </tr>
  <tr>
    <td>JtT</td>
    <td>Upweight the loss of the misclassified training data by 100 times. Train with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 128 for 60 epochs.</td>
  </tr>
  <tr>
    <td>MaskTune</td>
    <td>Train with Adam optimizer with a learning rate of 1e-4, a weight decay of 1e-4, and a batch size of 128 for 1 epoch.</td>
  </tr>
  <tr>
    <td>LfF</td>
    <td>Train with SGD optimizer with a learning rate of 5e-3, a weight decay of 1e-4, and a batch size of 128 for 50 epochs. We set the GCE hyperparamter q to 0.7.</td>
  </tr>
  <tr>
    <td>SoftCon</td>
    <td>Train an auxiliary BagNet18 model with Adam optimizer with a learning rate of 1e-3 for 20 epochs. Then, it refines the Original model using Adam optimizer with a learning rate of 5e-5 and a batch size of 32 for 10 epochs. We set the temperature for the contrastive learning loss to 0.1, cross-entropy loss weight to 1, and the clipping hyperparameter to 50.</td>
  </tr>
  <tr>
    <td>FLAC</td>
    <td>Use the same BagNet18 auxiliary model as SoftCon. Then, we refine the Original model with Adam optimizer with a learning rate of 5e-5 for 20 epochs and set the FLAC loss weight to 1000.</td>
  </tr>
  <tr>
    <td>LC</td>
    <td>We refine the Original model with SGD optimizer with a learning rate of 5e-3, a weight decay of 1e-3, and a batch size of 128 for 50 epochs. In parallel, it trains an auxiliary ResNet50 model with a SGD optimizer with a learning rate of 5e-4. The logit correction hyperparameter is set to 1, GCE hyperparameter $q$ to 0.8, and temperature to 0.1.</td>
  </tr>
  <tr>
    <td>CnC</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-4, and a batch size of 128 for 10 epochs.</td>
  </tr>
  <tr>
    <td>RRR</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 128 for 10 epochs. We set the RRR loss weights to 200.</td>
  </tr>
  <tr>
    <td>GradMask</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 128 for 10 epochs.</td>
  </tr>
  <tr>
    <td>ActDiff</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 128 for 10 epochs. 0.1 is multiplied to the loss for the distance between masked and unmasked representations.</td>
  </tr>
  <tr>
    <td>GradIA</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 128 for 10 epochs.</td>
  </tr>
  <tr>
    <td>Bounding Box</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 128 for 10 epochs.</td>
  </tr>
</table>

### Biased CelebA
<table>
  <tr>
    <td>CRAYON-Attention</td>
    <td>Fine-tune the Original model for 10 epochs with a batch size of 64, using the Adam optimizer with a learning rate of 1e-5 and a weight decay of 1e-4. The hyperparameters &alpha; and &beta; are set to 5e7 and 1e6, respectively.</td>
  </tr>
  <tr>
    <td>CRAYON-Pruning</td>
    <td>We prune 1,871 irrelevant neurons and train the last layer for 50 epochs with a learning rate of 5e-6.</td>
  </tr>
  <tr>
    <td>CRAYON-Attention+Pruning</td>
    <td>Fine-tune the Original model for 10 epochs with a batch size of 64, using the Adam optimizer with a learning rate of 1e-5 and a weight decay of 1e-4. The hyperparameters &alpha; and &beta; are set to 5e7 and 1e6, respectively.</td>
  </tr>
  <tr>
    <td>JtT</td>
    <td>Upweight the loss of the misclassified training data by 20 times. Trains the model with the Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-1, and a batch size of 64 for 30 epochs.</td>
  </tr>
  <tr>
    <td>MaskTune</td>
    <td>Train the Original model with Adam optimizer with a learning rate of 1e-9, a weight decay of 1e-4, and a batch size of 64 for 1 epoch.</td>
  </tr>
  <tr>
    <td>LfF</td>
    <td>Train the Original model with SGD optimizer with a learning rate of 5e-2, a weight decay of 1e-4, and a batch size of 64 for 50 epochs. We set the GCE hyperparamter q to 0.7.</td>
  </tr>
  <tr>
    <td>SoftCon</td>
    <td>Train an auxiliary BagNet18 model with Adam optimizer with a learning rate of 1e-3 for 20 epochs. Then, it refines the Original model using Adam optimizer with a learning rate of 5e-5 and a batch size of 32 for 10 epochs. We set the temperature for the contrastive learning loss to 0.1, cross-entropy loss weight &alpha; to 1, and the clipping hyperparameter &gamma; to 50.</td>
  </tr>
  <tr>
    <td>FLAC</td>
    <td>Use the same BagNet18 auxiliary model as SoftCon. We refine the Original model with Adam optimizer with a learning rate of 5e-5 for 5 epochs and set the FLAC loss weight of 1000.</td>
  </tr>
  <tr>
    <td>LC</td>
    <td>Refine the Original model with SGD optimizer with a learning rate of 1e-3, a weight decay of 1e-3, and a batch size of 64 for 50 epochs.
    In parallel, it trains an auxiliary ResNet50 model with a SGD optimizer with a learning rate of 1e-4.
    The logit correction hyperparameter &eta; is set to 1,
    GCE hyperparameter q to 0.8, and temperature to 0.1.</td>
  </tr>
  <tr>
    <td>CnC</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-1, and a batch size of 64 for 5 epochs.</td>
  </tr>
  <tr>
    <td>RRR</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-6, a weight decay of 1e-4, and a batch size of 64 for 5 epochs. We set the RRR loss weights to 25000.</td>
  </tr>
  <tr>
    <td>GradMask</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-6, a weight decay of 1e-4, and a batch size of 64 for 10 epochs.</td>
  </tr>
  <tr>
    <td>ActDiff</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-5, a weight decay of 1e-4, and a batch size of 64 for 10 epochs. 1e-5 is multiplied to the loss for the distance between masked and unmasked representations.</td>
  </tr>
  <tr>
    <td>GradIA</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-3, a weight decay of 1e-4, and a batch size of 64 for 10 epochs.</td>
  </tr>
  <tr>
    <td>Bounding Box</td>
    <td>Train the model with Adam optimizer with a learning rate of 5e-6, a weight decay of 1e-4, and a batch size of 64 for 10 epochs.</td>
  </tr>
</table>

### Backgrounds Challenge
<table>
  <tr>
    <td>CRAYON-Attention</td>
    <td>Fine-tune the classifier for 10 epochs with a batch size of 256 using the SGD optimizer with a learning rate of 5e-6 and a weight decay of 1e-1. The hyperparameters &alpha; and &beta; set to 5000 and 500, respectively.</td>
  </tr>
  <tr>
    <td>CRAYON-Pruning</td>
    <td>407 irrelevant neurons are pruned, and the last layer is trained for 10 epochs with a learning rate of 1e-6.</td>
  </tr>
  <tr>
    <td>CRAYON-Attention+Pruning</td>
    <td>For Backgrounds Challenge, we set &alpha; to 1000 and &beta; to 50. We use the SGD optimizer with a learning rate of 5e-5 and weight decay to 1e-1.</td>
  </tr>
  <tr>
    <td>JtT</td>
    <td>Upweight the loss of the misclassified training data by 5 times. Train the model with the Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-1, and a batch size of 256 for 10 epochs.</td>
  </tr>
  <tr>
    <td>MaskTune</td>
    <td>Train the Original model with Adam optimizer with a learning rate of 1e-7, a weight decay of 1e-5, and a batch size of 256 for 1 epoch.</td>
  </tr>
  <tr>
    <td>LfF</td>
    <td>Train the Original model with SGD optimizer with a learning rate of 1e-4, a weight decay of 1e-1, and a batch size of 256 for 1 epochs. We set the GCE hyperparamter q to 0.7.</td>
  </tr>
  <tr>
    <td>SoftCon</td>
    <td>Train an auxiliary BagNet18 model with Adam optimizer with a learning rate of 1e-3 for 20 epochs. Then, it refines the Original model using Adam optimizer with a learning rate of 5e-5 and a batch size of 128 for 10 epochs. We set the temperature for the contrastive learning loss to 0.07, cross-entropy loss weight &alpha; to 1e4, and the clipping hyperparameter &gamma; to 0.</td>
  </tr>
  <tr>
    <td>FLAC</td>
    <td>Use the same BagNet18 auxiliary model as SoftCon. We refine the Original model with Adam optimizer with a learning rate of 5e-6, a weight decay of 0.1, and a batch size of 128 for 5 epochs and set the FLAC loss weight of 100.</td>
  </tr>
  <tr>
    <td>LC</td>
    <td>Refine the Original model with SGD optimizer with a learning rate of 1e-4, a weight decay of 1e-1, and a batch size of 256 for 10 epochs.
    In parallel, it trains an auxiliary ResNet50 model with a SGD optimizer with a learning rate of 1e-5.
    The logit correction hyperparameter &eta; is set to 1,
    GCE hyperparameter q to 0.8, and temperature to 0.1.</td>
  </tr>
  <tr>
    <td>CnC</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-1, and a batch size of 256 for 10 epochs.</td>
  </tr>
  <tr>
    <td>RRR</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-4, and a batch size of 256 for 10 epochs. We set the RRR loss weights to 0.1.</td>
  </tr>
  <tr>
    <td>GradMask</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-4, and a batch size of 256 for 10 epochs.</td>
  </tr>
  <tr>
    <td>ActDiff</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-4, and a batch size of 256 for 10 epochs. 1e-2 is multiplied to the loss for the distance between masked and unmasked representations.</td>
  </tr>
  <tr>
    <td>GradIA</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-6, a weight decay of 1e-4, and a batch size of 256 for 5 epochs.</td>
  </tr>
  <tr>
    <td>Bounding Box</td>
    <td>Train the model with Adam optimizer with a learning rate of 1e-5, a weight decay of 1e-4, and a batch size of 256 for 10 epochs.</td>
  </tr>
</table>
