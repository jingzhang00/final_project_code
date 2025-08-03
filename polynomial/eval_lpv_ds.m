function F = eval_lpv_ds(X, ds_gmm, A_cell, x_star)
  [d, N] = size(X);
  K = numel(A_cell);
  R = zeros(K,N);
  for j=1:K
    R(j,:) = ds_gmm.Priors(j)*mvnpdf(X', ds_gmm.Mu(:,j)', ds_gmm.Sigma(:,:,j))';
  end
  R = R ./ (sum(R,1)+eps);
  F = zeros(d,N);
  for j=1:K
    F = F + A_cell{j}*(X - x_star).*R(j,:);
  end
end