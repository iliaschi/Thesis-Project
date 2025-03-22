1. Pre-training of a lightweight model on face identifi
cation task using very large facial dataset of celebri
ties [1].
 2. Fine-tuning the model from item (1) on static photos
 fromexternal dataset to obtain an emotional CNN [19].
 3. The outputs of the emotional CNN (embeddings and
 expression scores) from item (2) are used to extract
 facial features of each video frame from the AffWild2
 dataset [14].
 4. These embeddinsg and scores are used to train simple
 frame-level MLP-based classification/regression mod
els given the training set of each challange.
 5. Optional post-processing of frame-level outputs on
 models from item (4) computed for validation and test
 sets to make the predictions more smooth.
 Let us consider the details of our approach. At first, a
 large external VGGFace2 facial dataset [1] with 9131 sub
jects is used to pre-train a CNN on face recognition task.
 The faces cropped by MTCNN (multi-task cascaded neural
 network) [31] detector without any margins were utilized
 for training, so that most parts of the background, hairs,
 etc. is not presented. As a result, the learned facial fea
tures are more suitable for emotional analysis. We trained
 the model totally of 8 epochs by the Adam optimizer and
SAM (Sharpness-Aware Minimization) [6]. The models
 with the highest accuracy on validation set, namely, 92.1%,
 94.19%and95.49%forMobileNet-v1, EfficientNet-B0 and
 EfficientNet-B2, respectively, were used.
 Second, the resulted CNN is fine-tuned on the training
 set of 287,651 photos from the AffectNet dataset [19] an
notated with C = 8 basic expressions (Anger, Contempt,
 Disgust, Fear, Happiness, Neutral, Sadness and Surprise).
 It is necessary to emphasize that the annotations of valence
 and arousal from the AffectNet dataset were not used in the
 pre-training. The last layer of the network pre-trained on
 VGGFace2 is replaced by the new head (fully-connected
 layer with C outputs and softmax activation), so that the
 penultimate layer with D neurons can be considered as an
 extractor of facial features. The weighted categorical cross
entropy (softmax) loss was optimized [19]. The new head
 was trained during 3 epochs with learning rate 0.001. Fi
nally, the whole network is fine-tuned with a learning rate
 of 0.0001 at the last 5 epochs. The details of this training
 procedure are available in [24]. As a result, we fine-tuned
 three models, namely, MobileNet-v1, EfficientNet-B0 and
 EfficientNet-B2, that reached accuracy 60.71%, 61.32%
 and 63.03%, on the validation part of the AffectNet.
 Third, such an emotional CNN was used as a feature ex
tractor for frames X(t) and reference images Xn. Though
 the cropped facial images provided by the organizers of
 the challenge have different (typically, low) resolution, they
 were resized to 224x224 pixels for the first two models,
 while the latter CNN requires input images with resolution
 300x300. We examine two types of features: (1) facial im
age embeddings (output of penultimate layer) [24,29]; and
 (2) scores (predictions of emotional class probabilities at the
 output of last softmax layer). As a result, D-dimensional
 embeddings x(t) andxn andC-dimensionalscoress(t)and
 sn are obtained. Three kinds of features have been exam
ined, namely: (1) embeddings only; (2) scores only; and (3)
 concatenation of embeddings and scores [21]. According to
 the rules of the uni-task challenges, the pre-trained model
 can be pre-trained on any task (e.g., VA estimation, Ex
pression Classification, AU detection, Face Recognition),
 so that the expression scores returned by our model trained
 on the AffectNet can be used as facial features to predict Va
lence/Arousal and AUs. When we refined the model given
 the ABAW3 dataset, only the annotations available for a
 concrete challenge have been used to train a classification
 and regression models.
 Fourth, we trained a shallow feed-forward neural net
work, such as multi-class logistic regression or MLP (multi
layered perceptron) with one hidden layer for each of three
 tasks as follows:
 1. The output layer for expression recognition task con
tains CEXPR neurons with softmax activation. The
 weighted categorical cross-entropy was minimized for
 Figure 2. Sample screen of Android demo application.
 the first task. The final solution is taken in favor of fa
cial expression with the maximal predicted probability.
 2. Two neurons with tanh activations are used at the last
 layer to predict valence and arousal. The loss function
 is computed as 1 05(CCCV +CCCA)[15],where
 CCCV and CCCA are estimates of the Concordance
 Correlation Coefficient (CCC) for valence and arousal,
 respectively.
 3. Action unit detector contains CAU output units with
 sigmoid activation. The weighted binary cross-entropy
 loss is minimized. To predict the final binary vec
tor, the outputs of this model are matched with a
 f
 ixed threshold. We examine two possibilities, namely,
 one threshold (0.5) for each action unit or individ
ual threshold for each action unit. In the latter case,
 the best threshold is chosen from the list of 10 val
ues 0102 09 bymaximizingtheclass-levelF1
 score for the validation set.
 The model for each task is trained on 20 epochs with
 early stopping and Adam optimizer (learning rate 0.001).
 Fig. 1 contains the most general case of the proposed model
 with three outputs is trained for the multi-task learning chal
lenge. If the uni-task challenge is considered, only one out
put layer is used. Here, the facial regions are detected in
 each frame using MTCNN. The emotional features are ex
tracted using our EfficientNet model. These features are fed
 into MLP to solve one of the tasks or all tasks together in
 the multi-task learning scenario. If the facial region is not
 detected in a couple of frames, we perform the bilinear in
terpolation of the outputs of the model for two frames with
 detected faces. If face detection fails for several first or last
 frames of the video, we will simply use predictions for the
 closest frame with detected face.
 Fifth, it is possible to smooth the predictions for k 
 consecutive frames by using point-wisemean (box) or median filter 
 with kernel size k. If k i sequal to 1, the
 frame-level predictions will be used. Otherwise,the slicing
 window with size k is processed for every t-th frame, i.e.,
 we took the predictions at the output of our MLP classifiers
 for frames: t - k/2, t - (k/2) +1, ..., t + 1, ..., t + (k/2) - 1, t + (k/2).
 The final decision function for the frame t is computed as a
 point-wise mean or median of these k predictions.
 
 The training script for the presented approach is made
 publicly available1. The CNNs used for feature extrac
tion,namely,MobileNetv1(TensorFlow’smobilenet7.h5)
 and EfficientNets (PyTorch’s enetb08bestvgaf and
 enetb28),arealsoavailableinthisrepository2.Finally,the
 possibilitytouseourmodel formobiledevicesisdemon
strated.ThesampleoutputofthedemoAndroidapplication
 ispresentedinFig.2. It ispossibletorecognizefacialex
pressionsofallsubjectsineitheranyphotofromthegallery
 orthevideocapturedfromthefrontalcamera.