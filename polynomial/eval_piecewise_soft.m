function f = eval_piecewise_soft(x, piecewise_ds)
    K = numel(piecewise_ds);
    r = zeros(1,K);
    for i=1:K
        p = piecewise_ds{i}.Prior;
        mu = piecewise_ds{i}.Mu;
        S  = piecewise_ds{i}.Sigma;
        r(i) = p(i) * mvnpdf(x', mu', S);
    end
    r = r / sum(r); 

    f = zeros(size(x));
    for i=1:K
        pf = piecewise_ds{i}.f;
        phi = monolist(x, pf.var, pf.monom);
        f = f + r(i) * (pf.coeffs' * phi);
    end
end
