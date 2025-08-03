% function [poly_f, poly_V] = optimize_polynomial_lpv_ds(Xi_ref, Xi_dot_ref, opts)
%     import yalmip.*
% 
%     [M,N] = size(Xi_ref);
%     x = sdpvar(M,1);
% 
%     monom_f = monolist(x, opts.degree_f);
%     monom_V = monolist(x, opts.degree_V);
% 
%     c = sdpvar(length(monom_f), M, 'full');
%     v = sdpvar(length(monom_V), 1, 'full');
% 
%     f = c' * monom_f;
%     V = v' * monom_V;
% 
%     eps0 = opts.epsilon;
%     C1 = sos( V - eps0*(x'*x) );
%     LieV = jacobian(V,x)*f;
%     C2   = sos( -LieV - eps0*(x'*x) );
% 
%     Eq = replace(f, x, zeros(M,1)) == zeros(M,1);
% 
%     Fi = zeros(M,N);
%     for i = 1:N
%         Fi(:,i) = replace(f, x, Xi_ref(:,i));
%     end
%     Objective = sum(sum((Fi - Xi_dot_ref).^2));
% 
% 
%     % options = sdpsettings('solver','mosek','verbose', 0);
% 
%     options = sdpsettings( ...
%     'solver','bmibnb', ...        % 
%     'bmibnb.maxiter', 200, ...
%     'verbose',1, ...
%     'sos.model',2, ...           % more robust coefficient form
%     'sos.newton',1, ...          % Newton refinement on Gram matrix
%     'sos.congruence',1 ...       % enable congruence transforms
%     );
% 
%     decisionvars = [c(:); v(:)];
% 
%     degP1 = max(opts.degree_V, 2);
%     degP2 = max(opts.degree_V-1 + opts.degree_f, 2);
%     B1 = monolist(x, ceil(degP1/2));  
%     B2 = monolist(x, ceil(degP2/2));
%     basis = {B1, B2};
% 
%     AllC = [C1, C2, Eq];
%     [sol, u, Q] = solvesos(AllC, Objective, options, decisionvars, basis);
% 
%     if sol.problem ~= 0
%         disp('Sum-of-squares decomposition infeasible!');
%     end
% 
% 
% 
%     poly_f.coeffs  = value(c);
%     poly_f.degree  = opts.degree_f; 
%     poly_f.monom   = monom_f;
%     poly_f.dim     = M;
%     poly_f.var     = x;
% 
%     poly_V.coeffs  = value(v);
%     poly_V.degree  = opts.degree_V;
%     poly_V.monom   = monom_V;
% end

function [poly_f, poly_V] = optimize_polynomial_lpv_ds(Xc, Xdotc, opts)
    import yalmip.*

    [M,N] = size(Xc);
    s = sdpvar(1,1);
    x = sdpvar(M,1);
    alpha = 1e4;

    % 1) 符号 monomial basis
    monom_f = monolist(x, opts.degree_f);
    monom_V = monolist(x, opts.degree_V);

    Lf = length(monom_f);
    Lv = length(monom_V);


    c = sdpvar(Lf, M, 'full');
    v = sdpvar(Lv, 1, 'full');


    f = c' * monom_f;
    V = v' * monom_V;


    Eq = replace(f, x, zeros(M,1)) == zeros(M,1);


    Phi_f = zeros(Lf, N);
    for k = 1:N
        Phi_f(:,k) = double(replace(monom_f, x, Xc(:,k)));
    end

    eps0 = opts.epsilon;


    for iter = 1:opts.maxIter
        if iter == 1
            for dim = 1:M
                y = Xdotc(dim, :)';      % N×1
                A = Phi_f';              % N×Lf
                c0 = A \ y;              % Lf×1
                assign(c(:,dim), c0);
            end
            v0 = zeros(Lv,1);
            for j = 1:Lv
                mon = monom_V(j);
                if isequal(mon, x(1)^2) || isequal(mon, x(2)^2)
                    v0(j) = eps0;
                end
            end
            assign(v, v0);
        end


        C1_slack = sos( V - eps0*(x'*x) + s );
        if iter>1
            LieV  = jacobian(V,x)*f;
            C2c   = sos(-LieV - eps0*(x'*x));
            cons_c = [C1_slack, C2c, Eq];
        else
            cons_c = [C1_slack, Eq];
        end

        Objective_c = 0;
        for d = 1:M
            err = c(:,d)'*Phi_f - Xdotc(d,:);
            Objective_c = Objective_c + sum(err.^2);
        end

        Objective_c = Objective_c + alpha * s^2;


        cons_c = [cons_c, s >= 0];

        opts1 = sdpsettings('solver','sdpt3','sos.model',2,'verbose',0);
        solvesos(cons_c, Objective_c, opts1, [c(:); s]);


        f_fixed = replace(f, c, value(c));
        C1v = sos( V - eps0*(x'*x) + s );
        LieVv = jacobian(V,x)*f_fixed;
        C2v = sos(-LieVv - eps0*(x'*x));

        opts2 = sdpsettings('solver','sdpt3','sos.model',2,'verbose',0);
        solvesos([C1v, C2v, s >= 0], alpha*s^2, opts2, [v; s]);
    end


    poly_f.coeffs = value(c);
    poly_f.monom  = monom_f;
    poly_f.var    = x;
    poly_f.dim    = M;

    poly_V.coeffs = value(v);
    poly_V.monom  = monom_V;
    poly_V.var    = x;
    poly_V.dim    = M;
end

