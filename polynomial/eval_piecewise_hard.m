function F = eval_piecewise_hard(X, piecewise_ds)
    [M, N] = size(X);
    K = numel(piecewise_ds);

    R = zeros(K, N);
    for j = 1:K
        blk = piecewise_ds{j};
        mu = blk.Mu; 
        S  = blk.Sigma;
        p  = blk.Prior;
        R(j, :) = p * mvnpdf(X', mu', S)';
    end
    
    [~, labels] = max(R, [], 1);
    
    F = zeros(M, N);
    for j = 1:K
        idx = find(labels == j);
        if isempty(idx), continue; end
        
        blk = piecewise_ds{j};
        pf  = blk.f;
        

        Xj = X(:, idx); 
        Nj = size(Xj, 2);
        

        numMono = size(pf.coeffs, 1);
        PHI = zeros(numMono, Nj);
        
        % for k = 1:Nj
        %     xi = Xj(:, k);
        % 
        %     disp(['[DEBUG] block=', num2str(j), ...
        %           '  pt=', num2str(k),'/',num2str(Nj), ...
        %           '  xi size=', mat2str(size(xi)), ...
        %           '  degree=', num2str(pf.degree)]);
        % 
        % 
        %     phi_k = monolist(xi, pf.degree);
        % 
        % 
        %     if length(phi_k) ~= numMono
        %         error(['[ERROR] monolist returned ', num2str(length(phi_k)), ...
        %                ' terms but expect ', num2str(numMono)]);
        %     end
        % 
        %     PHI(:, k) = phi_k;
        % end
        

        F(:, idx) = pf.coeffs' * PHI;
    end
end
