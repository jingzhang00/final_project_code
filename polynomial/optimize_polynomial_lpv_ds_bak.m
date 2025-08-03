function [poly_f, poly_V] = optimize_polynomial_lpv_ds(Xi_ref, Xi_dot_ref, opts)
    import yalmip.*

    [M,N] = size(Xi_ref);
    x = sdpvar(M,1);

    monom_f = monolist(x, opts.degree_f);
    monom_V = monolist(x, opts.degree_V);

    c = sdpvar(length(monom_f), M, 'full');
    v = sdpvar(length(monom_V), 1, 'full');

    f = c' * monom_f;
    V = v' * monom_V;

    eps0 = opts.epsilon;
    C1 = sos( V - eps0*(x'*x) );
    LieV = jacobian(V,x)*f;
    C2   = sos( -LieV - eps0*(x'*x) );

    Eq = replace(f, x, zeros(M,1)) == zeros(M,1);

    Fi = zeros(M,N);
    for i = 1:N
        Fi(:,i) = replace(f, x, Xi_ref(:,i));
    end
    Objective = sum(sum((Fi - Xi_dot_ref).^2));


    options = sdpsettings('solver','mosek','verbose', 0);

    decisionvars = [c(:); v(:)];
    
    degP1 = max(opts.degree_V, 2);
    degP2 = max(opts.degree_V-1 + opts.degree_f, 2);
    B1 = monolist(x, ceil(degP1/2));  
    B2 = monolist(x, ceil(degP2/2));
    basis = {B1, B2};

    AllC = [C1, C2, Eq];
    [sol, u, Q] = solvesos(AllC, Objective, options, decisionvars, basis);

    if sol.problem ~= 0
        disp('Sum-of-squares decomposition infeasible!');
    end

    poly_f.coeffs = value(c);
    poly_f.monom = monom_f;
    poly_f.dim = M;
    poly_f.var = x;

    poly_V.coeffs = value(v);
    poly_V.monom = monom_V;
end

