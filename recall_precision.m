function [recall, precision, ap] = recall_precision(WTT, Dhamm)
% 计算召回率和精确率 (Precision and Recall)
% WTT: 地面真值相似矩阵 (WTT(i,j)=1 if query i and database j are similar)
% Dhamm: 汉明距离矩阵 (Dhamm(i,j) is the Hamming distance between query i and database j)

% [nq, n] = size(Dhamm);
% nq 是查询样本数量 (N_test), n 是数据库样本数量 (N_train)
[nq, n] = size(Dhamm); % 使用 Dhamm 的行数作为实际查询数量

% 检查 WTT 的行数是否大于 Dhamm 的行数 (可选的安全检查)
% if size(WTT, 1) > nq
%     fprintf('警告：WTT 的行数 (%d) 大于 Dhamm 的行数 (%d)。循环将仅执行 %d 次。\n', size(WTT, 1), nq, nq);
% end
% 重新定义 nq，确保它与 WTT 的行数一致

for i = 1:nq
    % 1. 对汉明距离进行升序排序，找到检索顺序
    [~, index] = sort(Dhamm(i, :), 'ascend');
    
    % 2. 依据检索顺序，获取该查询下的真实相似度 pos (1: 相似, 0: 不相似)
    pos = WTT(i, index) > 0;  % 正样本位置 (1 x n 向量)
    
    % 3. 计算累计正样本数 (Cumulative sum of positive samples)
    % cumsum(pos) 的长度为 n，表示在检索到第 k 个样本时，累计找到了多少个正样本。
    cumulative_pos = cumsum(pos);
    
    % 4. 计算召回率 (Recall)
    % recall = (累计正样本数) / (所有正样本总数)
    % sum(pos) 是该查询的所有正样本总数 (即 WTT(i,:) 中 1 的总数)
    total_relevant = sum(pos);
    if total_relevant == 0
        % 如果没有正样本，则跳过该查询的评估
        recall(i,:) = 0;
        precision(i,:) = 0;
        ap(i) = 0;
        continue;
    end
    recall(i,:) = cumulative_pos / total_relevant;
    
    % 5. 计算精确率 (Precision)
    % precision = (累计正样本数) / (已检索样本总数)
    % (1:length(pos))' 是已检索样本的总数 (1, 2, 3, ..., n)
    precision(i,:) = cumulative_pos ./ (1:length(pos));
    
    % 6. 计算平均精度 (Average Precision, AP)
    % 调用 area_RP 函数计算 P-R 曲线下的面积
    ap(i) = area_RP(recall(i,:), precision(i,:));
end

% 7. 对所有查询的结果取平均 (计算 Mean Average Precision 和平均 P-R 曲线)
recall = mean(recall, 1);
precision = mean(precision, 1);
ap = mean(ap); % 这就是最终的 mAP 值

end

% 注意：此文件依赖于 area_RP.m 函数来计算 P-R 曲线下的面积。