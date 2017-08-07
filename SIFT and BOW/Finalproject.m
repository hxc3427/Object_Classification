% Creation of the training set
trainingset=imageSet(fullfile('trainingset'));

% Creation of the testing set
testingset=imageSet(fullfile('testingset'));

% Creation of a function handle for SIFT feature extraction 
f=@computeSIFTfeatures; 

% Use of the feature extraction function handle to extract features from
% the training and testing set to create the visual vocabulary
bag=bagOfFeatures(trainingset,'CustomExtractor',f);

% Creation of the histogram recording the frequency of visual word 
% occurences for each image in the training set as their 
% corresponding feature vectors
d=zeros(trainingset.Count ,500);
for i=1:trainingset.Count
img=read(trainingset,i);
d(i,:) = double(encode(bag,img));
end

% Creation of the histogram recording the frequency of visual word 
% occurences for each image in the testing set as their corresponding 
% feature vectors
e=zeros(testingset.Count,500);
for b=1:testingset.Count
img=read(testingset,b);
e(b,:) = double(encode(bag,img));
end 

% Creation of the traininglabel and testinglabel vectors
traininglabel=double(zeros(trainingset.Count,1)); 
testinglabel=double(zeros(testingset.Count,1));

% Allocating labels in the training and testing label vectors
traininglabel(1:(trainingset.Count/3),1)=0;
traininglabel(((trainingset.Count/3)+1):(2*(trainingset.Count/3)),1)=1 ;
traininglabel(((2*(trainingset.Count/3))+1):(3*(trainingset.Count/3)),1)=2 ;
testinglabel(1:(testingset.Count/3),1)=0;
testinglabel(((testingset.Count/3)+1):(2*(testingset.Count/3)),1)=1;
testinglabel(((2*(testingset.Count/3))+1):(3*(testingset.Count/3)),1)=2;

% Training the SVM model on the training set
svm1=svmtrain1(traininglabel,d,'-c 1 -g 0.07');

% Testing the accuracy of the newly trained SVM model on the testing set
[predict_label, accuracy, prob_values] = svmpredict(testinglabel,e, svm1);
 
