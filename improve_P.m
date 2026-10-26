function P = improve_P( Xs, Xt, r)
    X_pca = [Xs, Xt];
    [U_pca, ~, ~] = svd(X_pca, 'econ');
    P = U_pca(:, 1:r);
 end
 
