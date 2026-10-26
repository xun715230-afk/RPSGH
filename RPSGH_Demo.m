clear all;
close all;
warning off;

addpath(genpath('./data')); 
addpath(genpath('./utils')); % clear all;
close all;
warning off;
rng(0);
addpath(genpath('./data')); 
addpath(genpath('./utils')); 
param.maxIter = 30; 
param.k = 25;        
param.bit = 64;
param.lambda1 = 1;
param.lambda2 = 10;
param.beta =0.1;
%  param.lambda_r=0;
%  MNIST-USPSS
% lambda1 =[0.1,1,10,100];   
% hash_reg =[1,10,100];
% 
%  USPS-MNIST
% lambda1 =[0.1,1,10,100];   
% hash_reg =[0.1,1,10];
% 
% COIL1-COIL2
% lambda1 =[0.1,1,10,100];
% hash_reg =[1,10,100];  
% 
% 
% Caltech256&ImageNet
% lambda1 =[1];
% hash_reg =[0.01]; 
% 
% Pr-Rw
% lambda1 =[0.1,1,10,100];
% hash_reg =[0.1,1,10,100]; 
% 
% A-D
% lambda1 =[0.1,1,10,100];
% hash_reg =[0.1,1,10,100]; 
% 
% D-A
% lambda1 =[0.1,1,10,100];
% hash_reg =[1,10]; 
% lambda1 =[100];
% hash_reg =[1]; 
test_num = 0.1;  
param.lambda_h = 0.1; 
%  tasks = {'Cl-Rw' , 'Rw-Cl','A-W','W-A' , 'Rw-Ar' , 'Ar-Rw' };
%            tasks = {'D-A','Pr-Rw', 'Rw-Pr','Cl-Rw' , 'Rw-Cl' };
          tasks = {'MNIST-USPS'};
%               tasks = {'Pr-Rw' };
%    tasks = {'Caltech256-ImageNet' };
%  tasks = { 'ImageNet-VOC2007', 'ImageNet-SUN09','ImageNet-LabelMe'};
%     'W-A' , 'Rw-Ar' , 'Cl-Rw' , 'Rw-Cl' ,'Pr-Rw', 'Rw-Pr' 
%    DB_name ='MNIST-USPS'   ;
    nbits = [16,32,48,64,96,128];
times = 1;
for db_idx = 1:length(tasks)
    DB_name = tasks{db_idx};
    for t = 1:times
        rng(0);
        dataset = construct_dataset(DB_name, test_num, param);
        [single_MAP, cross_MAP] = RPSGH_main(dataset, param);
        fprintf('task=%s, bit=%d, iter=%d, k=%d, single_MAP=%.2f, cross_MAP=%.2f\n', ...
            DB_name, param.bit, param.maxIter, param.k, single_MAP, cross_MAP);
    end
end


param.maxIter = 30; 
param.k = 25;        

tasks = {'D-A'};
test_num = 0.1;       
%% 

% 2. 锁定为 64 比特
nbits = [64];

% 3. 设定为跑 10 次
times = 10; 

% 这里的参数你可以填入你之前调试觉得最优的值
lambda1_list = [1]; 
hash_reg_list = [1];

%% ---------------------------------------------------------------
for db_idx = 1:length(tasks)
    DB_name = tasks{db_idx};
    fprintf('\n================ 开始测试任务: %s ================\n', DB_name);

    for b = 1:length(nbits) 
        param.bit = nbits(b); 

        for l1 = lambda1_list
            for hr = hash_reg_list   

               
                cross_MAP_list = zeros(1, times);
                single_MAP_list = zeros(1, times);

                param.lambda1 = l1;           
                param.hash_reg = hr;   

                for t = 1:times                                                 
       
                    rng(t); 
                    dataset = construct_dataset(DB_name, test_num, param);

                    % 接收输出结果到临时变量
                    [s_MAP, c_MAP] = RPSGH_main(dataset, param);

                    % 存入数组中
                    single_MAP_list(t) = s_MAP;
                    cross_MAP_list(t) = c_MAP;

                    fprintf('任务: %s | 64bits | 跑次第 %2d 次 | cross_MAP = %.2f\n', DB_name, t, c_MAP);      
                end

                % 计算 10 次的平均值和标准差
                mean_cmap = mean(cross_MAP_list);
                std_cmap = std(cross_MAP_list);

                fprintf('\n>>> 【最终结果汇总】%s (64 bits) | l1=%.2f, hash_reg=%.2f <<<\n', DB_name, param.lambda1, param.hash_reg);
                fprintf('>>> 10次平均 Cross-MAP: %.2f ± %.2f <<<\n\n', mean_cmap, std_cmap);

            end
        end
    end
end
% =========================================================
% RPSGH批量稳定评估脚本 (10次求均值与标准差)
% 包含任务: A-D, D-A, MNIST-USPS, USPS-MNIST, Pr-Rw, Rw-Pr
% 固定参数: lambda1 = 1.0, hash_reg = 1.0, 64 bits
% =========================================================
% clear all;
% close all;
% warning off;
% 
% % 添加必备的路径 (与原 Demo 保持一致)
% addpath(genpath('./data')); 
% addpath(genpath('./utils')); 
% 
% % 1. 全局参数设置
% param.maxIter = 30; 
% param.k = 25;        
% param.bit = 64;           % 锁定为 64 bits
% % param.lambda1 = 1.000000; % 锁定 l1 参数
% % param.hash_reg = 1.000000; % 锁定哈希回归正则参数
% param.lambda1 = 1;
% param.lambda2 = 10;
% param.beta =0.1;
% param.lambda_h = 0.1;
% test_num = 0.1;           % 测试集划分比例
% num_runs = 10;            % 每个任务跑 10 次
% 
% % 2. 定义需要遍历的 6 个任务列表
% % tasks = {'MNIST-USPS','A-D', 'D-A',  'USPS-MNIST', 'Pr-Rw', 'Rw-Pr','A-W','W-A' , 'Rw-Ar' , 'Ar-Rw','Cl-Rw' , 'Rw-Cl','ImageNet-LabelMe', 'ImageNet-VOC2007', 'ImageNet-SUN09','C1-C2','C2-C1' };
% % tasks = { 'Rw-Pr','A-W','W-A' , 'Rw-Ar' , 'Ar-Rw','Cl-Rw' , 'Rw-Cl','ImageNet-LabelMe', 'ImageNet-VOC2007', 'ImageNet-SUN09','C1-C2','C2-C1' };
% % tasks = {'ImageNet-LabelMe', 'ImageNet-VOC2007', 'ImageNet-SUN09'};
% % tasks = {'A-W','W-A' , 'Rw-Ar' , 'Ar-Rw','Cl-Rw' , 'Rw-Cl' ,'C1-C2','C2-C1','D-W','W-D'};
%   tasks = {'Caltech256-ImageNet','D-W','W-D' };
% %   tasks = {'D-W','W-D' };
% % 用于记录最终汇报表格的结构体
% final_report = struct();
% 
% fprintf('=======================================================\n');
% fprintf('启动 CDGM 批量评估程序\n');
% fprintf('统一参数: bits=%d, maxIter=%d, l1=%.2f, l2=%.2f,beta=%.2f,lh=%.2f\n', ...
%     param.bit, param.maxIter, param.lambda1, param.lambda2,param.beta,param.lambda_h);
% fprintf('执行计划: 共 %d 个任务，每个任务运行 %d 次...\n', length(tasks), num_runs);
% fprintf('=======================================================\n\n');
% 
% %% 3. 主循环遍历任务
% for db_idx = 1:length(tasks)
%     DB_name = tasks{db_idx};
%     fprintf('>>> 正在启动任务: [%s] <<<\n', DB_name);
% 
%     % 初始化当前任务的记录数组
%     single_MAP_list = zeros(1, num_runs);
%     cross_MAP_list  = zeros(1, num_runs);
% 
%     for t = 1:num_runs
%         % 【核心】使用 t 作为随机种子
%         % 这样每次 construct_dataset 划分的 test_num 都会不一样，
%         % 但整体 10 次实验是完全可复现的
%         rng(t); 
% 
%         % 加载并划分数据集
%         dataset = construct_dataset(DB_name, test_num, param);
% 
%         % 运行核心算法
%         [s_MAP, c_MAP, ~] = RPSGH_main(dataset, param);
% 
%         % 记录结果
%         single_MAP_list(t) = s_MAP;
%         cross_MAP_list(t)  = c_MAP;
% 
%         % 打印单次进度
%         fprintf('    Run %2d/%d | single_MAP = %.2f , cross_MAP = %.2f\n', ...
%             t, num_runs, s_MAP, c_MAP);
%     end
% 
%     % 4. 计算均值和标准差 (Std)
%     mean_single = mean(single_MAP_list);
%     std_single  = std(single_MAP_list);
%     mean_cross  = mean(cross_MAP_list);
%     std_cross   = std(cross_MAP_list);
% 
%     % 保存到汇报结构体中
%     safe_name =  strrep(DB_name, '-', '_');
%     final_report.(safe_name).mean_s = mean_single;
%     final_report.(safe_name).std_s  = std_single;
%     final_report.(safe_name).mean_c = mean_cross;
%     final_report.(safe_name).std_c  = std_cross;
% 
%     fprintf('--> [%s] 任务完成: Cross-MAP = %.2f ± %.2f\n\n', DB_name, mean_cross, std_cross);
% end
% 
% %% 5. 打印最终学术汇报表格
% fprintf('\n');
% fprintf('=======================================================\n');
% fprintf('                     10 次评估最终统计表格                 \n');
% fprintf('=======================================================\n');
% fprintf('%-15s | %-20s | %-20s\n', 'Dataset', 'Single-MAP (Mean±Std)', 'Cross-MAP (Mean±Std)');
% fprintf('-------------------------------------------------------\n');
% 
% for db_idx = 1:length(tasks)
%     DB_name = tasks{db_idx};
%     safe_name = strrep(DB_name, '-', '_');
% 
%     m_s = final_report.(safe_name).mean_s;
%     std_s = final_report.(safe_name).std_s;
%     m_c = final_report.(safe_name).mean_c;
%     std_c = final_report.(safe_name).std_c;
% 
%     str_single = sprintf('%.2f ± %.2f', m_s, std_s);
%     str_cross  = sprintf('%.2f ± %.2f', m_c, std_c);
% 
%     fprintf('%-15s | %-20s | %-20s\n', DB_name, str_single, str_cross);
% end
% fprintf('=======================================================\n');
% %% RPSGH_Demo_AllBenchmarks.m
% clear;
% clc;
% close all;
% 
% %% ==================== 1. 路径 ====================
% addpath(pwd);
% addpath(genpath('./data'));
% addpath(genpath('./utils'));
% 
% %% ==================== 2. 统一参数 ====================
% param.maxIter  = 30;
% param.k        = 25;
% param.lambda1  = 1;
% param.lambda2  = 10;
% param.beta     = 0.1;
% param.lambda_h = 0.1;
% 
% test_num = 0.1;      % 目标域10%作为测试查询集
% num_runs = 10;       % 每个任务、每个bit重复10次
% nbits = [16, 32, 48, 64, 96, 128];
% 
% % ==================== 3. 全部数据集 ====================
% % tasks = { ...
% %     'MNIST-USPS', ...
% %     'USPS-MNIST', ...
% %     'C1-C2', ...
% %     'C2-C1', ...
% %     'A-D', ...
% %     'A-W', ...
% %     'W-D', ...
% %     'D-A', ...
% %     'W-A', ...
% %     'D-W', ...
% %     'Pr-Rw', ...
% %     'Cl-Rw', ...
% %     'Rw-Ar', ...
% %     'Rw-Pr', ...
% %     'Rw-Cl', ...
% %     'Ar-Rw', ...
% %     'Caltech256-ImageNet', ...
% %     'ImageNet-LabelMe', ...
% %     'ImageNet-VOC2007', ...
% %     'ImageNet-SUN09' ...
% %     };
%  tasks = { 
%     'Pr-Rw'
%     };
% num_tasks = numel(tasks);
% num_bits  = numel(nbits);
% 
% %% ==================== 4. 保存所有原始结果 ====================
% % 维度：
% % task × bit × run
% 
% single_all = nan(num_tasks, num_bits, num_runs);
% cross_all  = nan(num_tasks, num_bits, num_runs);
% 
% fprintf('============================================================\n');
% fprintf('RPSGH Unified Benchmark\n');
% fprintf('Tasks      = %d\n', num_tasks);
% fprintf('Bits       = 16 32 48 64 96 128\n');
% fprintf('Runs       = %d\n', num_runs);
% fprintf('test_num   = %.2f\n', test_num);
% fprintf(['Parameters: maxIter=%d, k=%d, lambda1=%g, ' ...
%          'lambda2=%g, beta=%g, lambda_h=%g\n'], ...
%     param.maxIter, param.k, ...
%     param.lambda1, param.lambda2, ...
%     param.beta, param.lambda_h);
% 
% fprintf('Paper metric = Cross-MAP\n');
% fprintf('Target query -> Source retrieval set\n');
% fprintf('============================================================\n\n');
% 
% total_tic = tic;
% 
% %% ==================== 5. 正式实验 ====================
% for db_idx = 1:num_tasks
% 
%     DB_name = tasks{db_idx};
% 
%     fprintf('\n============================================================\n');
%     fprintf('[%2d/%2d] Task: %s\n', ...
%         db_idx, num_tasks, DB_name);
%     fprintf('============================================================\n');
% 
%     for b_idx = 1:num_bits
% 
%         param.bit = nbits(b_idx);
% 
%         fprintf('\n------ %d bits ------\n', param.bit);
% 
%         for run_idx = 1:num_runs
% 
%             %% ---------------------------------------------------------
%             % 随机性控制
%             %
%             % 同一个 run_idx：
%             % 各bit使用相同的数据划分
%             %
%             % 不同 run_idx：
%             % 使用不同数据划分和初始化
%             %
%             % 因此：
%             % 1. 10次实验确实不同
%             % 2. 不同bit之间又具有公平可比性
%             %% ---------------------------------------------------------
% 
%             split_seed = 10000 * db_idx + run_idx;
%             model_seed = 1000000 + 10000 * db_idx + run_idx;
% 
%             %% 目标域数据划分
%             rng(split_seed, 'twister');
% 
%             dataset = construct_dataset( ...
%                 DB_name, test_num, param);
% 
%             %% 模型内部随机初始化
%             rng(model_seed, 'twister');
% 
%             [single_MAP, cross_MAP, ~] = ...
%                 RPSGH_main(dataset, param);
% 
%             %% 保存
%             single_all(db_idx, b_idx, run_idx) = ...
%                 single_MAP;
% 
%             cross_all(db_idx, b_idx, run_idx) = ...
%                 cross_MAP;
% 
%             %% 显示当前实验
%             fprintf(['Run %2d/%2d | ' ...
%                      'single_MAP = %7.2f | ' ...
%                      'cross_MAP = %7.2f\n'], ...
%                 run_idx, num_runs, ...
%                 single_MAP, cross_MAP);
%         end
% 
%         %% 当前bit的10次统计
%         mean_single_now = ...
%             mean(single_all(db_idx,b_idx,:),3);
% 
%         std_single_now = ...
%             std(single_all(db_idx,b_idx,:),0,3);
% 
%         mean_cross_now = ...
%             mean(cross_all(db_idx,b_idx,:),3);
% 
%         std_cross_now = ...
%             std(cross_all(db_idx,b_idx,:),0,3);
% 
%         fprintf('\n');
% 
%         fprintf(['>>> %s | %3d bits | ' ...
%                  'Single = %.2f +/- %.2f | ' ...
%                  'Cross = %.2f +/- %.2f <<<\n'], ...
%             DB_name, ...
%             param.bit, ...
%             mean_single_now, ...
%             std_single_now, ...
%             mean_cross_now, ...
%             std_cross_now);
% 
%         %% 防止长时间实验中途丢失
%         save('RPSGH_all_results_checkpoint.mat', ...
%             'tasks', ...
%             'nbits', ...
%             'num_runs', ...
%             'test_num', ...
%             'param', ...
%             'single_all', ...
%             'cross_all');
%     end
% end
% 
% %% ==================== 6. 10次实验均值与标准差 ====================
% 
% single_mean = mean(single_all,3);
% single_std  = std(single_all,0,3);
% 
% cross_mean = mean(cross_all,3);
% cross_std  = std(cross_all,0,3);
% 
% %% 六种bit的总体平均
% % 对应图4里面的 Average
% 
% single_avg6 = mean(single_mean,2);
% cross_avg6  = mean(cross_mean,2);
% 
% %% ==================== 7. 打印论文Cross-MAP结果 ====================
% 
% fprintf('\n\n');
% fprintf('==========================================================================\n');
% fprintf('Cross-MAP: Mean MAP over 10 runs (%%)\n');
% fprintf('Target-domain Query -> Source-domain Retrieval Set\n');
% fprintf('==========================================================================\n');
% 
% fprintf('%-24s','Dataset');
% 
% for b_idx = 1:num_bits
%     fprintf('%10d',nbits(b_idx));
% end
% 
% fprintf('%12s\n','Average');
% 
% fprintf('%s\n', ...
%     repmat('-',1,24+10*num_bits+12));
% 
% for db_idx = 1:num_tasks
% 
%     fprintf('%-24s',tasks{db_idx});
% 
%     for b_idx = 1:num_bits
% 
%         fprintf('%10.2f', ...
%             cross_mean(db_idx,b_idx));
% 
%     end
% 
%     fprintf('%12.2f\n', ...
%         cross_avg6(db_idx));
% end
% 
% %% ==================== 8. Single-MAP也保留 ====================
% 
% fprintf('\n\n');
% fprintf('==========================================================================\n');
% fprintf('Single-MAP: Mean MAP over 10 runs (%%)\n');
% fprintf('Target Query -> Target Training Retrieval Set\n');
% fprintf('==========================================================================\n');
% 
% fprintf('%-24s','Dataset');
% 
% for b_idx = 1:num_bits
%     fprintf('%10d',nbits(b_idx));
% end
% 
% fprintf('%12s\n','Average');
% 
% fprintf('%s\n', ...
%     repmat('-',1,24+10*num_bits+12));
% 
% for db_idx = 1:num_tasks
% 
%     fprintf('%-24s',tasks{db_idx});
% 
%     for b_idx = 1:num_bits
% 
%         fprintf('%10.2f', ...
%             single_mean(db_idx,b_idx));
% 
%     end
% 
%     fprintf('%12.2f\n', ...
%         single_avg6(db_idx));
% end
% 
% %% ==================== 9. 生成论文CSV ====================
% 
% % 论文表格保留两位小数
% cross_mean_csv  = round(cross_mean,2);
% cross_std_csv   = round(cross_std,2);
% 
% single_mean_csv = round(single_mean,2);
% single_std_csv  = round(single_std,2);
% 
% cross_avg6_csv  = round(cross_avg6,2);
% single_avg6_csv = round(single_avg6,2);
% 
% bit_var_names = cell(1,num_bits);
% 
% for b_idx = 1:num_bits
% 
%     bit_var_names{b_idx} = ...
%         sprintf('Bit_%d',nbits(b_idx));
% 
% end
% 
% %% Cross-MAP
% T_cross_mean = ...
%     table(tasks(:), ...
%     'VariableNames',{'Dataset'});
% 
% T_cross_std = ...
%     table(tasks(:), ...
%     'VariableNames',{'Dataset'});
% 
% %% Single-MAP
% T_single_mean = ...
%     table(tasks(:), ...
%     'VariableNames',{'Dataset'});
% 
% T_single_std = ...
%     table(tasks(:), ...
%     'VariableNames',{'Dataset'});
% 
% for b_idx = 1:num_bits
% 
%     T_cross_mean.(bit_var_names{b_idx}) = ...
%         cross_mean_csv(:,b_idx);
% 
%     T_cross_std.(bit_var_names{b_idx}) = ...
%         cross_std_csv(:,b_idx);
% 
%     T_single_mean.(bit_var_names{b_idx}) = ...
%         single_mean_csv(:,b_idx);
% 
%     T_single_std.(bit_var_names{b_idx}) = ...
%         single_std_csv(:,b_idx);
% 
% end
% 
% T_cross_mean.Average_6bits = ...
%     cross_avg6_csv;
% 
% T_single_mean.Average_6bits = ...
%     single_avg6_csv;
% 
% %% 全部结果
% writetable( ...
%     T_cross_mean, ...
%     'RPSGH_CrossMAP_Mean_10Runs.csv');
% 
% writetable( ...
%     T_cross_std, ...
%     'CDGM_CrossMAP_Std_10Runs.csv');
% 
% writetable( ...
%     T_single_mean, ...
%     'RPSGH_SingleMAP_Mean_10Runs.csv');
% 
% writetable( ...
%     T_single_std, ...
%     'RPSGH_SingleMAP_Std_10Runs.csv');
% 
% %% ==================== 10. 按你的图片分别生成表格 ====================
% 
% %% 图1
% % MNIST->USPS
% % USPS->MNIST
% % COIL1->COIL2
% % COIL2->COIL1
% 
% writetable( ...
%     T_cross_mean([1 2 3 4],1:7), ...
%     'CDGM_Table_Fig1_Digit_COIL.csv');
% 
% %% 图2 Office31
% % A->D
% % A->W
% % W->D
% % D->A
% % W->A
% % D->W
% 
% writetable( ...
%     T_cross_mean([5 6 7 8 9 10],1:7), ...
%     'RPSGH_Table_Fig2_Office31.csv');
% 
% %% 图3 Office-Home
% % Pr->Rw
% % Cl->Rw
% % Rw->Ar
% % Rw->Pr
% % Rw->Cl
% % Ar->Rw
% 
% writetable( ...
%     T_cross_mean([11 12 13 14 15 16],1:7), ...
%     'RPSGH_Table_Fig3_OfficeHome.csv');
% 
% %% 图4
% % Caltech256->ImageNet
% % MNIST->USPS
% % 带 Average
% 
% writetable( ...
%     T_cross_mean([17 1],:), ...
%     'RPSGH_Table_Fig4_Caltech_ImageNet_MNIST_USPS.csv');
% 
% %% 图6
% % ImageNet->LabelMe
% % ImageNet->VOC2007
% % ImageNet->SUN09
% 
% writetable( ...
%     T_cross_mean([18 19 20],1:7), ...
%     'RPSGH_Table_Fig6_ImageNet_CrossDataset.csv');
% 
% %% ==================== 11. 保存1200次原始实验 ====================
% 
% N = num_tasks * num_bits * num_runs;
% 
% DatasetCol = cell(N,1);
% BitCol     = zeros(N,1);
% RunCol     = zeros(N,1);
% SingleCol  = zeros(N,1);
% CrossCol   = zeros(N,1);
% 
% row = 0;
% 
% for db_idx = 1:num_tasks
% 
%     for b_idx = 1:num_bits
% 
%         for run_idx = 1:num_runs
% 
%             row = row + 1;
% 
%             DatasetCol{row} = ...
%                 tasks{db_idx};
% 
%             BitCol(row) = ...
%                 nbits(b_idx);
% 
%             RunCol(row) = ...
%                 run_idx;
% 
%             SingleCol(row) = ...
%                 single_all(db_idx,b_idx,run_idx);
% 
%             CrossCol(row) = ...
%                 cross_all(db_idx,b_idx,run_idx);
% 
%         end
%     end
% end
% 
% T_all_runs = table( ...
%     DatasetCol, ...
%     BitCol, ...
%     RunCol, ...
%     SingleCol, ...
%     CrossCol, ...
%     'VariableNames', ...
%     {'Dataset', ...
%      'Bit', ...
%      'Run', ...
%      'Single_MAP', ...
%      'Cross_MAP'});
% 
% writetable( ...
%     T_all_runs, ...
%     'RPSGH_AllRuns.csv');
% 
% %% ==================== 12. 保存MAT ====================
% 
% save( ...
%     'RPSGH_AllBenchmark_Results.mat', ...
%     'tasks', ...
%     'nbits', ...
%     'num_runs', ...
%     'test_num', ...
%     'param', ...
%     'single_all', ...
%     'cross_all', ...
%     'single_mean', ...
%     'single_std', ...
%     'cross_mean', ...
%     'cross_std', ...
%     'single_avg6', ...
%     'cross_avg6');
% 
% elapsed_sec = toc(total_tic);
% 
% fprintf('\n');
% fprintf('============================================================\n');
% fprintf('ALL EXPERIMENTS FINISHED\n');
% fprintf('Total time = %.2f hours\n', ...
%     elapsed_sec/3600);
% fprintf('============================================================\n');
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              