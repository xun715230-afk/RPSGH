function dataset = construct_dataset(DB_name, test_num, param)
addpath('./data/');
addpath('./data/cross-dataset');
addpath('./data/VLSC');
addpath('./data/Office31');
addpath('./data/Office-Home(vgg)');
switch DB_name
    case 'MNIST-USPS'        
        load ./data/MNIST_vs_USPS X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
    case 'USPS-MNIST'        
        load USPS_vs_MNIST X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
        
    case 'VOC2007-Caltech101'        
        load VOC2007 data;
        X_src = double(normalize1(data(:, 1:4096)));       
        Y_src = double(data(:, 4097));
        clear data;
        
        load Caltech101 data;
        X_tar = double(normalize1(data(:, 1:4096)));
        Y_tar = double(data(:, 4097));
        clear data
                
    case 'Caltech256-ImageNet'        
        load dense_caltech256_decaf7_subsampled fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load dense_imagenet_decaf7_subsampled fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
                
    case 'Pr-Rw'
        load ./data/Office-Home(vgg)/Product_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7  fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Rw-Pr'        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/Product_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Cl-Rw'        
        load ./data/Office-Home(vgg)/Clipart_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Rw-Cl'        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/Clipart_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Ar-Rw'        
        load ./data/Office-Home(vgg)/Art_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'Rw-Ar'        
        load ./data/Office-Home(vgg)/RealWorld_vgg16_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        clear fts labels;
        
        load ./data/Office-Home(vgg)/Art_vgg16_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        clear fts labels;
        
    case 'P27-P05'        
        load PIE27 fea gnd;
        X_src = double(normalize1(fea));
        Y_src = double(gnd);
        
        load PIE05 fea gnd;
        X_tar = double(normalize1(fea));
        Y_tar = double(gnd);
        
    case 'A-W'
        load amazon_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load webcam_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
       
    case 'A-D'
        load ./data/Office31/amazon_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load ./data/Office31/dslr_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
    
    case 'W-D'
        load webcam_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load dslr_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        
    case 'D-A'
        load dslr_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load amazon_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
       
    case 'W-A'
        load webcam_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load amazon_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
    
    case 'D-W'
        load dslr_fc7 fts labels;
        X_src = double(normalize1(fts));
        Y_src = double(labels);
        
        load webcam_fc7 fts labels;
        X_tar = double(normalize1(fts));
        Y_tar = double(labels);
        
    case 'C1-C2'        
        load ./data/COIL_1 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);
                
    case 'C2-C1'        
        load COIL_2 X_src X_tar Y_src Y_tar;
        X_src = double(normalize1(X_src'));
        X_tar = double(normalize1(X_tar'));
        Y_src = double(Y_src);
        Y_tar = double(Y_tar);    

        case 'ImageNet-LabelMe'
        % 注意：这里的 'ImageNet' 和 'LabelMe' 需要与你本地 ./data 文件夹里的 .mat 文件名完全一致
        load ImageNet data; 
        X_src = double(normalize1(data(:, 1:4096))); % 假设特征是 4096 维
        Y_src = double(data(:, 4097));
        clear data;
        
        load LabelMe data;
        X_tar = double(normalize1(data(:, 1:4096)));
        Y_tar = double(data(:, 4097));
        clear data;

    case 'ImageNet-VOC2007'
        load ImageNet data;
        X_src = double(normalize1(data(:, 1:4096)));
        Y_src = double(data(:, 4097));
        clear data;
        
        load VOC2007 data;
        X_tar = double(normalize1(data(:, 1:4096)));
        Y_tar = double(data(:, 4097));
        clear data;

    case 'ImageNet-SUN09'
        load ImageNet data;
        X_src = double(normalize1(data(:, 1:4096)));
        Y_src = double(data(:, 4097));
        clear data;
        
        load SUN09 data;
        X_tar = double(normalize1(data(:, 1:4096)));
        Y_tar = double(data(:, 4097));
        clear data;
  
end

% 统一标签为 1..C（避免非连续/非1起始标签导致的错误）
Y_src = Y_src(:);
Y_tar = Y_tar(:);
all_labels = [Y_src; Y_tar];
[~, ~, mapped_labels] = unique(all_labels, 'stable');
Y_src = mapped_labels(1:numel(Y_src));
Y_tar = mapped_labels(numel(Y_src)+1:end);

c = length(unique(Y_tar));  % The number of classes
dataset.c = c;

%% 分层划分测试集（按类别比例抽样）
classes = unique(Y_tar);
test_idx = [];
train_idx = [];
if test_num < 1
    for ii = 1:length(classes)
        cls = classes(ii);
        idx = find(Y_tar == cls);
        cnt = numel(idx);
        if cnt == 0
            continue;
        end
        n_test = round(test_num * cnt);
        rp = idx(randperm(cnt));
        test_idx = [test_idx; rp(1:n_test)];
        train_idx = [train_idx; rp(n_test+1:end)];
    end
else
    total = min(test_num, numel(Y_tar));
    counts = arrayfun(@(cls) sum(Y_tar == cls), classes);
    ratio = counts / sum(counts);
    raw = total * ratio;
    n_test_per = floor(raw);
    remain = total - sum(n_test_per);
    [~, order] = sort(raw - n_test_per, 'descend');
    for ii = 1:remain
        n_test_per(order(ii)) = n_test_per(order(ii)) + 1;
    end
    for ii = 1:length(classes)
        cls = classes(ii);
        idx = find(Y_tar == cls);
        cnt = numel(idx);
        if cnt == 0
            continue;
        end
        rp = idx(randperm(cnt));
        n_test = min(n_test_per(ii), cnt);
        test_idx = [test_idx; rp(1:n_test)];
        train_idx = [train_idx; rp(n_test+1:end)];
    end
end

Xt_test = X_tar(test_idx, :);
Yt_test = Y_tar(test_idx);
Xt = X_tar(train_idx, :);
Yt = Y_tar(train_idx);
nt = numel(train_idx);

% ���Լ�
dataset.Xt_test = Xt_test;
dataset.Yt_test = Yt_test;
% Retain the exact target-domain partition for reproducible exports.
dataset.target_test_indices = test_idx;
dataset.target_train_indices = train_idx;
% ʣ��һ����Ŀ�����ȫ��Դ��������Ϊѵ����
dataset.Xs = X_src;
dataset.Ys = Y_src;
dataset.Xt = Xt;
dataset.Yt = Yt;

dataset.nt = nt;
dataset.ns = size(X_src, 1);

    
X=[dataset.Xs;dataset.Xt];
samplemean = mean(X,1);
dataset.Xs = dataset.Xs-repmat(samplemean,size(dataset.Xs,1),1);
dataset.Xt = dataset.Xt-repmat(samplemean,size(dataset.Xt,1),1);
dataset.Xt_test = dataset.Xt_test-repmat(samplemean,size(dataset.Xt_test,1),1);


dataset.d = size(dataset.Xs, 2);



%% ����PR����
YS = repmat(Y_src, 1, length(Yt_test));     
YT = repmat(Yt_test, 1, length(Y_src));
WTT = (YT==YS');
dataset.WTT = WTT;

YT = repmat(Yt, 1, length(Yt_test));
YTest = repmat(Yt_test,1,length(Yt));
WTT_single = (YTest==YT');
dataset.WTT_single = WTT_single;

end
