function [S, S_ss, S_st] = Init_S_Structure(dataset, Pt, k, P, O)
    Xs = dataset.Xs;
    Xt = dataset.Xt;
    Ys = dataset.Ys(:);
    c  = dataset.c;
    nt = dataset.nt;
    ns = dataset.ns;

    Ys_onehot = full(sparse(1:ns, double(Ys), 1, ns, c));
    S_ss = double(Ys_onehot * Ys_onehot' > 0);

% Projected features
Us = Xs * P;
Ut = Xt * P;
% O_norm  =O;
Us_norm = normalize_rows(Us);
Ut_norm = normalize_rows(Ut);
O_norm  = normalize_rows(O);

P_proto = softmax_rows(Ut_norm * O_norm');      
Q_t = sqrt(Pt .* P_proto + eps);                
Sim_st = Us_norm * Ut_norm';                    
Src_proto_sim = sum(Us_norm .* O_norm(Ys, :), 2);  
Score = bsxfun(@plus, Sim_st, Src_proto_sim);      
S_st = zeros(ns, nt);
for cls = 1:c
    idx = (Ys == cls);
    A_cls = softmax_cols(Score(idx, :));           
    S_st(idx, :) = bsxfun(@times, A_cls, Q_t(:, cls)');
end
S_st = bsxfun(@rdivide, S_st, sum(S_st, 1) + eps);

[Pt_graph] = Build_GPO_Pt_Graph(Pt);
    Xt_knn = normalize_rows(Xt);
    D_raw = EuDist2_local(Xt_knn, Xt_knn);
    k_nn = min(k, nt);
    [~, Idx] = sort(D_raw, 2, 'ascend');

    S_tt = zeros(nt, nt);
    for i = 1:nt
        id = Idx(i, 1:k_nn);
        S_tt(i, id) = (Pt_graph(id, :) * Pt_graph(i, :)')';
    end
  S_tt = bsxfun(@rdivide, S_tt, sum(S_tt,1) + eps);
    
  S_tt_sem = S_st' * S_st + Q_t * Q_t';
    mu =  norm(S_tt_sem, 'fro') / (norm(S_tt, 'fro') + eps);
    S = [S_ss, S_st; S_st',  mu*S_tt];

end

function Xn = normalize_rows(X)
    Xn = bsxfun(@rdivide, X, sqrt(sum(X.^2, 2)) + eps);
end

function P = softmax_rows(S)
    S = bsxfun(@minus, S, max(S, [], 2));
    E = exp(S);
    P = bsxfun(@rdivide, E, sum(E, 2) + eps);
end

function P = softmax_cols(S)
    S = bsxfun(@minus, S, max(S, [], 1));
    E = exp(S);
    P = bsxfun(@rdivide, E, sum(E, 1) + eps);
end

function D = EuDist2_local(A, B)
    nA = sum(A.^2, 2);
    nB = sum(B.^2, 2);
    D = bsxfun(@plus, nA, nB') - 2 * A * B';
    D = max(D, 0);
end
function [Pt_graph] = Build_GPO_Pt_Graph(Pt)
 P = bsxfun(@rdivide, ...
        Pt, sum(Pt, 2) + eps);

    %  类别组GPO：消除类别公共响应
    E = bsxfun(@minus, ...
        P, mean(P, 1));
    Z = bsxfun(@rdivide, ...
        E, sqrt(mean(E.^2, 1) + eps));

    %  样本组GPO：突出样本内部优势类别
    A = bsxfun(@minus, ...
        Z, mean(Z, 2));
    A = bsxfun(@rdivide, ...
        A, sqrt(mean(A.^2, 2) + eps));

    % 正优势连续分配
    A = max(A, 0);
    A = bsxfun(@rdivide, ...
        A, sum(A, 2) + eps);
    %  原始正证据 × 相对GPO优势
    Pt_graph = max(E, 0) .* A;

end


  