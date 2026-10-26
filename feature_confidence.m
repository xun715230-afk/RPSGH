function T = feature_confidence(P, X, n, mode)
if nargin < 4
    mode = 'proposed';
end

% The ablation removes only the feature-confidence weighting.
if strcmpi(mode, 'zero')
    T = sparse(n, n);
    return;
end
if strcmpi(mode, 'uniform')
    T = speye(n);
    return;
end

Z = P' * X;
Z = bsxfun(@minus, Z, mean(Z, 2));
u = mean(abs(Z), 1)';
u = u / (mean(u) + eps);
v = mean((u - 1).^2);

lu = log(u + eps);
 mu = mean(lu);
s2 = mean((lu - mu).^2);
t_diag = (u ./ (1 + v*u)) .* exp(-((lu - mu).^2) / ((s2 + eps) * (1 + v)));
t_diag = t_diag / (mean(t_diag) + eps);
T = spdiags(t_diag, 0, n, n);
end
