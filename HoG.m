clc;
clear all;
run('/Users/harshdeep/Downloads/libsvm-3.21/matlab/make.m');
%% ETRACTING FEATURES FOR TRAINING SET
Path = '/Users/harshdeep/Desktop';
setDir = fullfile(Path,'1');
imgSets = imageSet(setDir, 'recursive');
[trainingSets, testSets] = partition(imgSets, 0.25, 'randomize');
Total_class=3;
% Classfeaturevector=cell(Total_class,1);
trainingFeatures = [];
trainingLabels   = [];
for j= 1: length(trainingSets)
for i = 1 : trainingSets(j).Count
        TrainIMG= read(trainingSets(j),i);
        TrainIMG= imresize(TrainIMG,[40 40]);
        TrainIMG= mat2gray(TrainIMG);
        [featureVector, hogVisualization] = extractHOGFeatures(TrainIMG,'CellSize',[4 4]);
        X(i,:) = featureVector;
%        if j==1
%         X(i,:) = featureVector;
%        end
%        if j==2
%            X1(i,:)= featureVector;
%        end
%        if j==3
%            X3(i,:)= featureVector;
%        end
end
     labels = repmat(trainingSets(j).Description, trainingSets(j).Count, 1);
     trainingFeatures = [trainingFeatures;X];  
     trainingLabels   = [trainingLabels;labels];
end
%% EXTRACTING FEATURE FOR TEST SET
testFeatures = [];
testLabels   = [];
for j= 1: length(testSets)
for i = 1 : testSets(j).Count
        TestIMG= read(testSets(j),i);
        TestIMG= imresize(TestIMG,[40 40]);
        TestIMG= mat2gray(TestIMG);
        [featureVector1, hogVisualization1] = extractHOGFeatures(TestIMG,'CellSize',[4 4]);
       Y(i,:) = featureVector1;
%        if j==1
%         X(i,:) = featureVector1;
%        end
%        if j==2
%            X1(i,:)= featureVector;
%        end
%        if j==3
%            X3(i,:)= featureVector;
%        end
end
     labels = repmat(testSets(j).Description, testSets(j).Count, 1);
     testFeatures = [testFeatures;Y];  
     testLabels   = [testLabels;labels];
end
%% TRAINING CLASSIFIER AND CHECKING ACCURACY
% TrainClasifer= svmtrain(trainingFeatures,trainingLabels,'showplot',true,'Kernel_Function','rbf');
% Model = svmtrain(trainingFeatures,trainingLabels);
% [predictedLabels,accuracy,~] = svmpredict(TestLab,double(TestSet),Model);
 classifier = fitcecoc(trainingFeatures, trainingLabels);
predictedLabels = predict(classifier, testFeatures);
C = confusionmat(testLabels,predictedLabels);
Accuracy = sum(diag(C))/sum(sum(C));
disp('The Accuracy Of the classifier using HOG Features extraction is');
disp(Accuracy*100);