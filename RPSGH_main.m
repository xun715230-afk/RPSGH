function [single_MAP,cross_MAP,param_out] = RPSGH_main(dataset, param)
Xs = dataset.Xs';
Ys = dataset.Ys';
ns = dataset.ns;
Xt = dataset.Xt';
nt = dataset.nt;
n = ns + nt;
X_test = dataset.Xt_test';
c = dataset.c;
maxIter = param.maxIter;
r = param.bit;
k = param.k;
lambda1 = param.lambda1;
 lambda2 = param.lambda2;
 lambda_h = param.lambda_h;
beta  = param.beta ;
gamma = 5;
Ys = Ys(:);
%% Initialization
X = [Xs, Xt];
Ys_onehot = dummyvar(Ys(:));
P = improve_P(Xs, Xt,r);
[~, O] = kmeans((P' * X)', c);
O = bsxfun(@rdivide, O, sqrt(sum(O.^2, 2)) + eps) * sqrt(r);
T = feature_confidence(P, X, n);
[Lt, Pt] = updateLt(P, Xs, Xt,  O, Ys, c);
Y = Build_Label_Matrix(Ys,Lt, n, c);

U = P' * X;
 C_hash = eye(r);
[S,S_ss, S_st] = Init_S_Structure(dataset, Pt, k, P,  O);
 B = Update_B_From_Graph_Prototype(U, S, T, C_hash, beta, lambda2, r);
%% Optimization
for iter = 1:maxIter
    M_1 = Compute_Robust_Trace_Matrices(P, Xs, Xt, Ys_onehot, ns, nt, k);
    P = Update_P_Core(X, M_1, O, lambda1, gamma, Y, T);
    T = feature_confidence(P, X, n);
    O = Update_O(X, P, Y, T);

[Lt, Pt] = updateLt(P, Xs, Xt, O, Ys, c);
Y = Build_Label_Matrix(Ys, Lt, n, c);
U = P' * X;
[S,S_ss, S_st] = Init_S_Structure(dataset, Pt, k, P, O);
C_hash = Fit_Code_Prototype_Map(B, U, S, T, beta, lambda2, r);
B = Update_B_From_Graph_Prototype(U, S, T, C_hash, beta, lambda2, r);
end
%% Evaluation
Bs = B(:, 1:ns);
Bt = B(:, ns+1:end);
H = Fit_Ridge_Hash_Map(B, X, lambda_h);
Bq = Quantize_Code(H * X_test);

B_db_cross  = Bs';
B_db_single = Bt';
B_query     = Bq';

B_db_cross(B_db_cross == 0)     = 1;
B_db_single(B_db_single == 0)   = 1;
B_query(B_query == 0)           = 1;



D_cross = hammingDist(compactbit(B_query > 0), compactbit(B_db_cross > 0));
cross_MAP = local_map_from_distance(dataset.WTT, D_cross);

D_single = hammingDist(compactbit(B_query > 0), compactbit(B_db_single > 0));
single_MAP = local_map_from_distance(dataset.WTT_single, D_single);

param_out = param;
end
%%
function MAP = local_map_from_distance(Wtrue, Dist)
    nq = size(Dist, 1);
    ap_sum = 0;
    valid_q = 0;
    for ii = 1:nq
        [~, idx_rank] = sort(Dist(ii, :), 'ascend');
        rel = Wtrue(ii, idx_rank) > 0;
        n_rel = sum(rel);
        if n_rel == 0
            continue;
        end
        prec_at_k = cumsum(rel) ./ (1:length(rel));
        ap_sum = ap_sum + sum(prec_at_k(rel)) / n_rel;
        valid_q = valid_q + 1;
    end
    if valid_q == 0
        MAP = 0;
    else
        MAP = 100 * ap_sum / valid_q;
    end
end

function P = Update_P_Core(X, M_1, O, lambda1, gamma, Y, T)
    [d, ~] = size(X);
    LHS = X * (lambda1 * M_1 + T) * X' + gamma * eye(d);
    RHS = X * T * Y * O;
    P = LHS \ RHS;
end

function B = Quantize_Code(Z)
    B = sign(Z);
    B(B == 0) = 1;
end

function Y = Build_Label_Matrix(Ys, Lt, n,c )
     Y = full(sparse(1:n, [Ys(:); Lt(:)], 1, n, c));

end

function C_hash = Fit_Code_Prototype_Map(B, U, S, T, beta, lambda2, r)

    A_u = U * U';
    M_b = B * B';
    D_align = U*T;
    G = beta * A_u + lambda2 * eye(r);

    Q = beta * r * U * S' * B' ...
      + lambda2 * D_align * B';

    mb_rcond = rcond(M_b);
    g_rcond = rcond(G);
    is_bad_system = ~isfinite(mb_rcond) || ~isfinite(g_rcond) || ...
        (mb_rcond * g_rcond < 1e-12);

    if is_bad_system
        M_b_reg = stabilize_hash_gram(M_b);
        C_hash = (G \ Q) / M_b_reg;
    else
        K = kron(M_b', G);
        C_hash = reshape(K \ Q(:), r, r);
    end

    if any(~isfinite(C_hash(:)))
        M_b_reg = stabilize_hash_gram(M_b);
        C_hash = (G \ Q) / M_b_reg;
    end

    [Uc, ~, Vc] = svd(C_hash, 'econ');
    C_hash = Uc * Vc';
end

function M_reg = stabilize_hash_gram(M)
    M_sym = (M + M') / 2;
    eig_vals = eig(M_sym);
    max_eig = max(eig_vals);
    min_eig = min(eig_vals);
    if ~isfinite(max_eig) || max_eig <= 0
        max_eig = 1;
    end
    if ~isfinite(min_eig)
        min_eig = 0;
    end

    target_rcond = 1e-8;
    ridge = max(target_rcond * max_eig - min_eig, eps(max_eig));
    M_reg = M_sym + ridge * eye(size(M, 1));
end

function B = Update_B_From_Graph_Prototype(U, S, T, C_hash, beta, lambda2, r)
    graph_term = beta * r .* C_hash' * U * S';
    align_term = lambda2 * C_hash' * U * T;
    B = Quantize_Code(graph_term + align_term);
   
end

function H = Fit_Ridge_Hash_Map(B, Z, lambda_h)
    d = size(Z, 1);
    H = (B * Z') / (Z * Z' + lambda_h * eye(d));
end

function O = Update_O(X, P, Y, T)
% Scaled rectangular Procrustes update (no numerical change).
% c <= r: O*O' = r*I_c; c >= r: O'*O = r*I_r.
% It exactly maximizes trace((Y'*T*X'*P)'*O) over the
% dimension-adaptive scaled partial-isometry constraint.
    [~, r] = size(P);
    A = Y' *T * X' * P;
    [OU, ~, OV] = svd(A, 'econ');
    O = sqrt(r) * (OU * OV');
end

function M_1 = Compute_Robust_Trace_Matrices(P, Xs, Xt, Ys_onehot, ns, nt, k)
    n = ns + nt;
    norm_factor = nt;
    Xs_proj = P' * Xs;
    Xt_proj = P' * Xt;
    dist_matrix = pdist2(Xt_proj', Xs_proj');
    lambda_weights = zeros(nt, 1);

    for j = 1:nt
        [sorted_dists, sorted_idx] = sort(dist_matrix(j, :));
        knn_idx = sorted_idx(1:k);
        knn_dists = sorted_dists(1:k);
        knn_labels = Ys_onehot(knn_idx, :);
        q_j = max(sum(knn_labels, 1));
        w_j = q_j / k;

        d_min = knn_dists(1);
        d_max = knn_dists(end);
        if d_max < eps
            r_j = 1;
        else
            r_j = d_min / d_max;
        end
        lambda_weights(j) = w_j * r_j;
    end

    M_1 = zeros(n, n);
    for j = 1:nt
        M_1(ns+j, ns+j) = k * lambda_weights(j) / norm_factor;
    end

    source_diag = zeros(ns, 1);
    for j = 1:nt
        [~, sorted_idx] = sort(dist_matrix(j, :));
        knn_idx = sorted_idx(1:k);
        for i = knn_idx
            source_diag(i) = source_diag(i) + lambda_weights(j) / norm_factor;
        end
    end

    for i = 1:ns
        M_1(i, i) = source_diag(i);
    end

    for j = 1:nt
        [~, sorted_idx] = sort(dist_matrix(j, :));
        knn_idx = sorted_idx(1:k);
        for i = knn_idx
            M_1(i, ns+j) = -lambda_weights(j) / norm_factor;
            M_1(ns+j, i) = -lambda_weights(j) / norm_factor;
        end
    end
end

function [Lt, Pt] = updateLt(P, Xs, Xt, O, Ys, c)
    domainS_proj = (P' * Xs)';
    domainT_proj = (P' * Xt)';

%     源域类中心（O(i,:) 作为第 cnt+1 个伪样本融入
    r = size(domainS_proj, 2);
    srcMeans = zeros(c, r);
    for i = 1:c
        cnt = sum(Ys == i);
        srcMeans(i, :) = (sum(domainS_proj(Ys == i, :), 1) + O(i, :)) / (cnt + 1);
    end

%     目标域聚类
    [~, tarMeans] = kmeans(domainT_proj, c, 'Start', srcMeans, 'MaxIter', 100, 'Replicates', 1);

%     两路概率
    prob1 = softmax_rows(EuDist2(domainT_proj, srcMeans));
    prob2 = softmax_rows(EuDist2(domainT_proj, tarMeans));

%     ---------- 融合 ----------
    probFused = (prob1+ prob2)/2;   
    Pt=probFused ;
    [~, Lt] =  max(probFused, [], 2);
    Lt = Lt';
end
function prob = softmax_rows(dist)
    d = bsxfun(@minus, dist, min(dist, [], 2));
    e = exp(-d);
    prob = bsxfun(@rdivide, e, sum(e, 2) + eps);
end
function D = EuDist2(A, B)
    nA = sum(A.^2, 2);
    nB = sum(B.^2, 2);
    D = bsxfun(@plus, nA, nB') - 2 * A * B';
    D = max(D, 0);
end





