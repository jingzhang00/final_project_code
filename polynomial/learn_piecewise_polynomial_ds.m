function [piecewise_ds, region_info] = learn_piecewise_polynomial_ds(Xi_ref, Xi_dot_ref, ds_gmm, att, sos_options)
    % Input:
    %   Xi_ref, Xi_dot_ref: reference trajectories
    %   ds_gmm: GMM structure with Mu, Sigma, Priors
    %   att: attractor point
    %   sos_options: polynomial degrees and options
    
    import yalmip.*
    
    [M, N] = size(Xi_ref);
    K = size(ds_gmm.Mu, 2); % Number of GMM components
    
    % Initialize piecewise DS structure
    piecewise_ds = struct();
    piecewise_ds.K = K;
    piecewise_ds.regions = cell(K, 1);
    piecewise_ds.local_poly = cell(K, 1);
    piecewise_ds.weights = cell(K, 1);
    
    region_info = struct();
    region_info.data_assignment = zeros(1, N);
    region_info.region_sizes = zeros(K, 1);
    
    fprintf('Learning piecewise polynomial DS with %d regions...\n', K);
    
    % Step 1: Assign data points to GMM regions (soft assignment)
    [responsibilities, hard_assignment] = assign_data_to_regions(Xi_ref, ds_gmm);
    region_info.data_assignment = hard_assignment;
    
    % Step 2: Learn polynomial DS for each region
    for k = 1:K
        fprintf('\n--- Learning polynomial for region %d ---\n', k);
        
        % Get data points for this region (using soft assignment)
        region_weights = responsibilities(k, :);
        region_threshold = 0.3; % Minimum responsibility to include data point
        region_indices = find(region_weights > region_threshold);
        
        if length(region_indices) < 10
            fprintf('Warning: Region %d has only %d data points, using hard assignment\n', k, length(region_indices));
            region_indices = find(hard_assignment == k);
        end
        
        if length(region_indices) < 5
            fprintf('Warning: Region %d has insufficient data, skipping\n', k);
            continue;
        end
        
        Xi_region = Xi_ref(:, region_indices);
        Xi_dot_region = Xi_dot_ref(:, region_indices);
        region_weights_selected = region_weights(region_indices);
        
        region_info.region_sizes(k) = length(region_indices);
        
        % Center data around region center for numerical stability
        region_center = ds_gmm.Mu(:, k);
        Xi_region_centered = Xi_region - region_center;
        
        % Store region information
        piecewise_ds.regions{k}.center = region_center;
        piecewise_ds.regions{k}.cov = ds_gmm.Sigma(:, :, k);
        piecewise_ds.regions{k}.prior = ds_gmm.Priors(k);
        piecewise_ds.regions{k}.data_indices = region_indices;
        
        % Learn local polynomial DS for this region
        try
            local_sos_options = sos_options;
            local_sos_options.degree_f = min(sos_options.degree_f, 3); % Limit degree for stability
            local_sos_options.degree_V = min(sos_options.degree_V, 2);
            local_sos_options.epsilon = max(sos_options.epsilon, 1e-2);
            
            [poly_f_local, poly_V_local] = learn_local_polynomial_ds(...
                Xi_region_centered, Xi_dot_region, att - region_center, ...
                local_sos_options, region_weights_selected);
            
            piecewise_ds.local_poly{k} = poly_f_local;
            fprintf('Successfully learned polynomial for region %d\n', k);
            
        catch ME
            fprintf('Failed to learn polynomial for region %d: %s\n', k, ME.message);
            
            % Fallback: simple linear DS for this region
            fprintf('Using linear fallback for region %d\n', k);
            [poly_f_local] = learn_linear_fallback(Xi_region_centered, Xi_dot_region, att - region_center);
            piecewise_ds.local_poly{k} = poly_f_local;
        end
        
        % Learn blending weights (Gaussian-like)
        piecewise_ds.weights{k} = @(x) compute_region_weight(x, ds_gmm.Mu(:, k), ds_gmm.Sigma(:, :, k));
    end
    
    fprintf('\nPiecewise polynomial DS learning completed!\n');
    
    % Create combined DS function
    piecewise_ds.evaluate = @(x) evaluate_piecewise_ds(x, piecewise_ds);
end

function [responsibilities, hard_assignment] = assign_data_to_regions(Xi_ref, ds_gmm)
    % Compute soft and hard assignments of data points to GMM regions
    
    [M, N] = size(Xi_ref);
    K = size(ds_gmm.Mu, 2);
    
    % Compute responsibilities (soft assignment)
    log_likelihoods = zeros(K, N);
    
    for k = 1:K
        diff = Xi_ref - ds_gmm.Mu(:, k);
        inv_cov = inv(ds_gmm.Sigma(:, :, k));
        mahal_dist = sum((inv_cov * diff) .* diff, 1);
        log_det_cov = log(det(ds_gmm.Sigma(:, :, k)));
        
        log_likelihoods(k, :) = log(ds_gmm.Priors(k)) - 0.5 * (M * log(2*pi) + log_det_cov + mahal_dist);
    end
    
    % Compute normalized responsibilities
    max_log_like = max(log_likelihoods, [], 1);
    log_likelihoods_stable = log_likelihoods - max_log_like;
    likelihoods = exp(log_likelihoods_stable);
    responsibilities = likelihoods ./ sum(likelihoods, 1);
    
    % Hard assignment
    [~, hard_assignment] = max(responsibilities, [], 1);
end

function [poly_f, poly_V] = learn_local_polynomial_ds(Xi_ref, Xi_dot_ref, att_local, opts, weights)
    % Learn polynomial DS for a local region
    
    import yalmip.*
    
    [M, N] = size(Xi_ref);
    
    % Define symbolic variables
    x = sdpvar(M, 1);
    
    % Create monomial bases with limited degrees
    monom_f = monolist(x, opts.degree_f);
    monom_V = monolist(x, opts.degree_V);
    
    % Define coefficient matrices
    c = sdpvar(length(monom_f), M, 'full');
    v = sdpvar(length(monom_V), 1, 'full');
    
    % Define polynomial functions
    f = c' * monom_f;
    V = v' * monom_V;
    
    % Constraints
    eps0 = opts.epsilon;
    
    % Lyapunov function constraints (relaxed for local regions)
    C1 = sos(V - eps0 * (x' * x));
    
    % Local stability constraint
    LieV = jacobian(V, x) * f;
    C2 = sos(-LieV - eps0 * (x' * x));
    
    % Local equilibrium constraint (towards local attractor)
    Eq = (replace(f, x, att_local) == zeros(M, 1));
    
    % Weighted data fitting objective
    Fi = zeros(M, N);
    for i = 1:N
        Fi(:, i) = replace(f, x, Xi_ref(:, i));
    end
    
    % Weighted objective
    if nargin >= 5 && ~isempty(weights)
        Objective = sum(sum((Fi - Xi_dot_ref).^2 .* weights));
    else
        Objective = sum(sum((Fi - Xi_dot_ref).^2));
    end
    
    % Try different solvers
    solvers_to_try = {'sedumi', 'sdpt3', 'mosek'};
    sol = [];
    
    for solver_idx = 1:length(solvers_to_try)
        current_solver = solvers_to_try{solver_idx};
        
        try
            options = sdpsettings('solver', current_solver, 'verbose', 0, ...
                                'sos.newton', 1, 'sos.congruence', 1);
            
            decisionvars = [c(:); v(:)];
            
            % Simplified basis for local problems
            degP1 = min(opts.degree_V, 2);
            degP2 = min(opts.degree_V - 1 + opts.degree_f, 3);
            B1 = monolist(x, ceil(degP1/2));
            B2 = monolist(x, ceil(degP2/2));
            basis = {B1, B2};
            
            AllC = [C1, C2, Eq];
            [sol, u, Q] = solvesos(AllC, Objective, options, decisionvars, basis);
            
            if sol.problem == 0
                break;
            end
        catch
            continue;
        end
    end
    
    if isempty(sol) || sol.problem ~= 0
        error('Local polynomial optimization failed');
    end
    
    % Extract results
    poly_f.coeffs = value(c);
    poly_f.monom = monom_f;
    poly_f.dim = M;
    poly_f.var = x;
    
    poly_V.coeffs = value(v);
    poly_V.monom = monom_V;
    poly_V.dim = M;
    poly_V.var = x;
end

function [poly_f] = learn_linear_fallback(Xi_ref, Xi_dot_ref, att_local)
    % Simple linear DS as fallback: dx = A*(x - att_local)
    
    import yalmip.*
    
    [M, N] = size(Xi_ref);
    
    % Define linear system: dx = A*x + b, with constraint that A*att_local + b = 0
    A = sdpvar(M, M, 'full');
    
    % Data fitting for linear system
    X_data = Xi_ref;
    Xdot_data = Xi_dot_ref;
    
    % Objective: minimize ||A*X - Xdot||^2
    Objective = sum(sum((A * X_data - Xdot_data).^2));
    
    % Constraint: A*att_local = 0 (equilibrium at local attractor)
    if norm(att_local) > 1e-6
        Constraint = (A * att_local == zeros(M, 1));
    else
        Constraint = [];
    end
    
    % Solve
    options = sdpsettings('solver', 'quadprog', 'verbose', 0);
    optimize(Constraint, Objective, options);
    
    A_opt = value(A);
    
    % Create polynomial structure (linear is degree 1)
    x = sdpvar(M, 1);
    monom_f = [ones(1,1); x]; % [1; x1; x2; ...]
    
    % Coefficients for f(x) = A*x
    c = zeros(length(monom_f), M);
    c(2:end, :) = A_opt'; % Skip constant term, fill in linear terms
    
    poly_f.coeffs = c;
    poly_f.monom = monom_f;
    poly_f.dim = M;
    poly_f.var = x;
end

function weight = compute_region_weight(x, center, cov)
    % Compute Gaussian weight for blending
    diff = x - center;
    inv_cov = inv(cov);
    mahal_dist = diff' * inv_cov * diff;
    weight = exp(-0.5 * mahal_dist);
end

function dx = evaluate_piecewise_ds(x, piecewise_ds)
    % Evaluate piecewise polynomial DS
    
    K = piecewise_ds.K;
    dx = zeros(size(x));
    total_weight = 0;
    
    for k = 1:K
        if isempty(piecewise_ds.local_poly{k})
            continue;
        end
        
        % Transform to local coordinates
        x_local = x - piecewise_ds.regions{k}.center;
        
        % Evaluate local polynomial
        try
            dx_local = evaluate_polynomial_ds(x_local, piecewise_ds.local_poly{k});
            
            % Compute blending weight
            weight = piecewise_ds.weights{k}(x);
            
            dx = dx + weight * dx_local;
            total_weight = total_weight + weight;
        catch
            continue;
        end
    end
    
    % Normalize
    if total_weight > 1e-6
        dx = dx / total_weight;
    end
end

function dx = evaluate_polynomial_ds(x, poly_f)
    % Evaluate polynomial dynamical system at point x
    import yalmip.*
    
    if size(x, 1) ~= poly_f.dim
        error('Input dimension mismatch');
    end
    
    % Evaluate monomials at x
    monom_vals = replace(poly_f.monom, poly_f.var, x);
    
    % Compute f(x) = c' * monom_vals
    dx = poly_f.coeffs' * double(monom_vals);
end