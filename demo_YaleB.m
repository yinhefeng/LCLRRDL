clear
clc
close all

addpath('lrr')
load('YaleB96x84.mat')

% parameters setting for LCLRRDL
% should be tuned on other datasets
param=[];
param.lambda=25;
param.alpha=0.5;
param.gamma=0.05;

% number of classes
ClassNum=length(unique(gnd));

% number of images for each subject
EachClassNum=zeros(1,ClassNum);
for i=1:ClassNum
    EachClassNum(i)=sum(i==gnd);
end

% number of experiments
experiments=5;
reg_rate=zeros(1,experiments);

% number of training samples and atoms for each subject
train_num=32;
basePerCls=20;

k=1;
for ii=1:experiments
    ii
    
    % randomly select training samples for each subject
    train_ind=[];
    for i=1:ClassNum
        sample_tol=EachClassNum(i);
        t_ind=zeros(1,sample_tol);
        temp=randperm(sample_tol);
        t_ind(temp(1:train_num))=1;
        train_ind=[train_ind,t_ind];
    end
    
    % indices for the training and test data
    train_ind=logical(train_ind);
    test_ind=~train_ind;
    
    % training data and their labels
    train_data=fea(:,train_ind);
    train_label=gnd(:,train_ind);
    
    % test data and their labels
    test_data=fea(:,test_ind);
    test_label=gnd(:,test_ind);
    
    % unit l2 norm
    train_data=normc(train_data);
    test_data=normc(test_data);
    
    % number of training and test samples
    train_tol=length(train_label);
    test_tol=length(test_label);
    
    % label matrix for training data
    H_train=full(ind2vec(train_label,ClassNum));
    
    % initialize the dictionary with randomly selected training samples
    D_ind=[];
    for i=1:ClassNum
        temp=zeros(1,train_num);
        randnum=randperm(train_num);
        temp(randnum(1:basePerCls))=1;
        D_ind=[D_ind,temp];
    end
    D_ind=logical(D_ind);
    Dinit=train_data(:,D_ind);
    
    % construct the weight matrix
    dist = L2_distance(Dinit,train_data);
    % normalize to [0,1]
    dist = exp(dist-repmat(max(dist,[],2),[1 size(dist,2)]));
    param.dist = dist;
    
    % low rank decomposition
    [Z_tr, D, E] = WLRR(train_data, Dinit, param);
    
    % obtain the representation of test data by LRR
    Z_te = inexact_alm_lrr_l1(test_data,D,0.1);
    
    %linear classifier by ridge regression
    lambda = 0.001;
    W=H_train*Z_tr'/(Z_tr*Z_tr'+lambda*eye(size(Z_tr*Z_tr')));
    [~,pre_label]=max(W*Z_te);
    
    reg_rate(k)=sum(pre_label==test_label)/test_tol
    k=k+1;
end

% report the average and standard deviation of recognition accuracy
mean(reg_rate)
std(reg_rate)

