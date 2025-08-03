clear all; close all; clc

% Select one of the motions from the LASA Handwriting Dataset
sub_sample      = 5; % Each trajectory has 1000 samples when set to '1'
nb_trajectories = 7; % Maximum 7, will select randomly if <7
model_names = {'Angle','BendedLine','CShape','DoubleBendedLine','GShape',...
    'heee','JShape','JShape_2','Khamesh','Leaf_1',...
    'Leaf_2','Line','LShape','NShape','PShape',...
    'RShape','Saeghe','Sharpc','Sine','Snake',...
    'Spoon','Sshape','Trapezoid','Worm','WShape','Zshape',...
    'Multi_Models_1', 'Multi_Models_2', 'Multi_Models_3','Multi_Models_4'};

modelList = [1, 3, 6, 11, 12, 15, 18, 19, 28];

set(0, 'DefaultFigureVisible', 'off');
nRep = 5;

metrics   = {'RMSE','Edot','DTWD_mean','DTWD_std','Time'};
for rep = 1:nRep
    nModels = numel(modelList);
    vals    = nan(numel(metrics), nModels);
    fprintf('=== Starting repetition %d/%d ===\n', rep, nRep);

    for m = 1:nModels
        i = modelList(m);
        close all force
        current_model_name = model_names{i};

        [Data, Data_sh, att, x0_all, ~, dt] = batch_load_LASA_dataset_DS(sub_sample, nb_trajectories, i);
        vel_samples = 15; vel_size = 0.5;
    
        [h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);
        
        
        fprintf('%%%%%%%%%%%%%%%%%%%%%% evaluation for: %s %%%%%%%%%%%%%%%%%%%%\n', current_model_name);
        axis_limits = axis;
        
        M = size(Data,1)/2;    
        Xi_ref = Data(1:M,:);
        Xi_dot_ref = Data(M+1:end,:);  
        tStart = cputime;
        
        est_options = [];
        est_options.type             = 0;   % GMM Estimation Algorithm Type
        
        est_options.samplerIter      = 50; 
                                            
        est_options.do_plots         = 1;
        
        nb_data = length(Data);
        sub_sample = 1;
        if nb_data > 500
            sub_sample = 2;
        elseif nb_data > 1000
                sub_sample = 3;
        end
        est_options.sub_sample       = sub_sample;       
        
        
        est_options.estimate_l       = 1;   % '0/1' Estimate the lengthscale, if set to 1
        est_options.l_sensitivity    = 2;   % lengthscale sensitivity [1-10->>100]
                                            % Default value is set to '2' as in the
                                            % paper, for very messy, close to
                                            % self-intersecting trajectories, we
                                            % recommend a higher value
        est_options.length_scale     = [];  % if estimate_l=0 you can define your own
                                            % l, when setting l=0 only
                                            % directionality is taken into account
        
        % Fit GMM to Trajectory Data
        [Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);
        
        
        [idx] = knnsearch(Mu', att', 'k', size(Mu,2));
        Priors = Priors(:,idx);
        Mu     = Mu(:,idx);
        Sigma  = Sigma(:,:,idx);
        
        Sigma(:,:,1) = 1.*max(diag(Sigma(:,:,1)))*eye(M);
        Mu(:,1) = att;
        
        clear ds_gmm; ds_gmm.Mu = Mu; ds_gmm.Sigma = Sigma; ds_gmm.Priors = Priors; 
        
        
        adjusts_C  = 1;
        if adjusts_C  == 1 
            if M == 2
                tot_dilation_factor = 1; rel_dilation_fact = 0.25;
            elseif M == 3
                tot_dilation_factor = 1; rel_dilation_fact = 0.75;        
            end
            Sigma_ = adjust_Covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact);
            ds_gmm.Sigma = Sigma_;
        end   
        
        
        [~, est_labels] =  my_gmm_cluster(Xi_ref, ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, 'soft', [0.4, 0.8]);
        
        [h_gmm]  = visualizeEstimatedGMM(Xi_ref,  ds_gmm.Priors, ds_gmm.Mu, ds_gmm.Sigma, est_labels, est_options);
        
        lyap_constr = 2;      % 0:'convex':     A' + A < 0 (Proposed in paper)
                              % 2:'non-convex': A'P + PA < -Q given P (Proposed in paper)                                 
        init_cvx    = 1;      % 0/1: initialize non-cvx problem with cvx 
        symm_constr = 0;      % This forces all A's to be symmetric (good for simple reaching motions)
        
        if lyap_constr == 0 || lyap_constr == 1
            P_opt = eye(M);
        else
            % P-matrix learning
            % [Vxf] = learn_wsaqf(Data,0,att);
           
            % (Data shifted to the origin)
            % Assuming origin is the attractor (works better generally)
            [Vxf] = learn_wsaqf(Data_sh);
            P_opt = Vxf.P;
        end
        

        %%%%%%%%  LPV system sum_{k=1}^{K}\gamma_k(xi)(A_kxi + b_k) %%%%%%%% 
        constr_type = 2;
        if constr_type == 1
            [A_k, b_k, P_est] = optimize_lpv_ds_from_data(Data, zeros(M,1), constr_type, ds_gmm, P_opt, init_cvx);
            ds_lpv = @(x) lpv_ds(x-repmat(att, [1 size(x,2)]), ds_gmm, A_k, b_k);
        else
            [A_k, b_k, ~] = optimize_lpv_ds_from_data(Data, att, constr_type, ds_gmm, P_opt, init_cvx);
            ds_lpv = @(x) lpv_ds(x, ds_gmm, A_k, b_k);
        end
        
        % This will be reported later as "loose" training time (visualizations make
        % it slower than what is actually necessary for training)
        tEnd = cputime - tStart;

        %% %%%%%%%%%%%%    Plot Resulting DS  %%%%%%%%%%%%%%%%%%%
        % Fill in plotting options
        ds_plot_options = [];
        ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
        ds_plot_options.x0_all    = x0_all;       % Intial Points
        ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
                                                  % 'ellipsoid' or 'cube'
        ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
        ds_plot_options.plot_vol  = 0;            % Plot volume of initial points (3D)
        ds_plot_options.save_path  = sprintf('%s_visualization.pdf', current_model_name);

        [hd, hs, hr, x_sim] = visualizeEstimatedDS(Xi_ref, ds_lpv, ds_plot_options);
        limits = axis;
        switch constr_type
            case 0
                title('GMM-based LPV-DS with QLF', 'Interpreter', 'LaTex', 'FontSize', 20)
            case 1
                title('GMM-based LPV-DS with P-QLF (v0) ', 'Interpreter', 'LaTex', 'FontSize', 20)
            case 2
                title('GMM-based LPV-DS with P-QLF', 'Interpreter', 'LaTex', 'FontSize', 20)
        end
        h_vel = visualizeEstimatedVelocities(Data, ds_lpv);

%% 

        clc
        disp('--------------------')
        
        % Compute RMSE on training data
        rmse = mean(rmse_error(ds_lpv, Xi_ref, Xi_dot_ref));
        fprintf('LPV-DS with (O%d), got velocity RMSE on training set: %d \n', constr_type+1, rmse);
        
        % Compute e_dot on training data
        edot = mean(edot_error(ds_lpv, Xi_ref, Xi_dot_ref));
        fprintf('LPV-DS with (O%d), got velocity deviation (e_dot) on training set: %d \n', constr_type+1, edot);
        
        % Display time 
        fprintf('DS trained in %1.2f seconds (only true for when you run the whole script).\n', tEnd);
        
        % Compute DTWD between train trajectories and reproductions
        if ds_plot_options.sim_traj
            nb_traj       = size(x_sim, 3);
            ref_traj_leng = size(Xi_ref, 2) / nb_traj;
            dtwd = zeros(1, nb_traj);
            for n=1:nb_traj
                start_id = round(1 + (n-1) * ref_traj_leng);
                end_id   = round(n * ref_traj_leng);
                dtwd(1,n) = dtw(x_sim(:,:,n)', Xi_ref(:,start_id:end_id)', 20);
            end
            fprintf('LPV-DS got DTWD of reproduced trajectories: %2.4f +/- %2.4f \n', mean(dtwd), std(dtwd));
        end

        vals(:,m) = [rmse; edot; mean(dtwd); std(dtwd); tEnd];
end
    T = array2table(vals, ...
         'VariableNames', model_names(modelList), ...
         'RowNames',      metrics);
    
    fname = sprintf('metrics_rep%02d.csv', rep);
    writetable(T, fname, 'WriteRowNames', true);
    fprintf('  → saved %s\n', fname);
end

