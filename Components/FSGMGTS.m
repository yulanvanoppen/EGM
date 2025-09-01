classdef FSGMGTS < handle
    properties (SetAccess = private)
        data                                                                % data aggregate struct
        system                                                              % nested object controlling the ODE system
        settings                                                            % hyperparameters and user input
        
        T                                                                   % number of time points
        L                                                                   % number of observed states
        N                                                                   % population size
        
        beta_fs                                                             % cell-specific parameter estimates
        beta_fs_history                                                     % record past iterations
        V                                                                   % spline/ode gradient error variances
        varXdX                                                              % state and gradient covariances
        varbeta                                                             % uncertainty estimates of beta
        
        convergence_steps                                                   % relative iteration steps
        not_converged                                                       % indices of cells that have not yet converged
        
        smoothed_fitted                                                     % combination of predicted (hidden)
        dsmoothed_fitted                                                    % and smoothed (observed) states/gradients)
        fitted_fs                                                           % predicted states and gradients
        dfitted_fs
    end
    
    
    methods
        function obj = FSGMGTS(data, system, settings, beta_init, x_init) % Constructor
            obj.data = data;                                                % experimental and smoothing data
            obj.data.init = x_init;
            obj.system = system;                                            % ODE system
            obj.settings = settings;                                        % hyperparameters
            
            [obj.T, obj.L, obj.N] = size(data.smoothed);                    % extract from trajectory dimensions
            
            obj.beta_fs = beta_init;
            obj.V = repmat(eye(system.K * obj.T), 1, 1, obj.N);
            obj.varXdX = zeros(2 * system.K * obj.T, 2 * system.K * obj.T, obj.N);
            obj.varbeta = repmat(eye(system.P), 1, 1, obj.N);
            
            obj.convergence_steps = ones(1, obj.N);                         % ensure no cells are considered converged
            obj.not_converged = 1:obj.N;
            
            obj.integrate(settings.niter == 0)                              % save solution if 'converged'
        end
        
        
        function output = optimize(obj)                                 % Main optimization function
            obj.smoothed_fitted(:, obj.data.observed, :) = obj.data.smoothed;
            obj.dsmoothed_fitted(:, obj.data.observed, :) = obj.data.dsmoothed;
            
            for iter = 1:obj.settings.niter
                fprintf('%d ', iter)

                beta_old = obj.beta_fs;

                if iter > 1, obj.integrate; end                             % ODE integration
                obj.estimate_covariances(iter);                             % residual covariance estimation
                obj.update_parameters();                                    % gradient matching

                if ~mod(iter, 10), fprintf('(%.3e)\n', eucl_rel(beta_old, obj.beta_fs)), end

                                                                            % break when 90% of cells have converged
                if eucl_rel(beta_old, obj.beta_fs) < obj.settings.tol || iter == obj.settings.niter
                    break
                end
            end
            
            obj.save_estimates();                                           % save estimates and fitted trajectories
            output = obj.data;                                              % return data appended with FS results
        end
            
        
        function update_parameters(obj)                                 % Update (cell-specific) parameter estimates
            for i = 1                                                       % model slopes from splines and integrations
                design = obj.system.g(obj.smoothed_fitted(2:end-1, :, i), obj.data.t(2:end-1));
                const = obj.system.h(obj.smoothed_fitted(2:end-1, :, i), obj.data.t(2:end-1));
                response = obj.dsmoothed_fitted(2:end-1, :, i) - const;     % spline and integration slopes
    
                                                                            % disregard interval ends
                nonzero_ind = reshape((2:obj.T-1)' + obj.T*(0:obj.system.K-1), 1, []);
                variances = obj.V(nonzero_ind, nonzero_ind, i);
                weights = flatten(sqrt(obj.settings.weights(2:end-1, :)));
                variances = variances ./ weights ./ weights';
                                                                            % constrained GLS using quadratic programming
                obj.beta_fs(i, :) = Optimization.QPGLS(design, response, variances, obj.settings.lb, ...
                                                        obj.settings.ub);
            end
        end
        
                                        
        function estimate_covariances(obj, iter, converged)             % Update error covariance function parameters
            if nargin < 3, converged = false; end 
            obj.covariances_partial(iter, converged);                       % partial observation
        end
        
        
        function integrate(obj, converged)                              % Hidden states integration
            if nargin == 1, converged = false; end
            
            obj.fitted_fs = obj.system.integrate(obj.beta_fs, obj.data);    % integrate system and compute rhs
            if obj.settings.positive
                obj.fitted_fs = max(1e-12, obj.fitted_fs);                  % force positive
            end
            obj.dfitted_fs = obj.system.rhs(obj.fitted_fs, obj.data.t, obj.beta_fs);
            unobserved = ~ismember(1:obj.system.K, obj.data.observed);      % replace unobserved states by ODE integrations
            
                                                                            % substitute integrated hidden states and gradients
            obj.smoothed_fitted(:, unobserved, :) = obj.fitted_fs(:, unobserved, :); 
            obj.dsmoothed_fitted(:, unobserved, :) = obj.dfitted_fs(:, unobserved, :);
            
            if converged                                                    % store smooth versions upon convergence
                obj.data.fitted_fs_fine = obj.system.integrate(obj.beta_fs, obj.data, obj.data.t_fine);
                obj.data.dfitted_fs_fine = obj.system.rhs(obj.data.fitted_fs_fine, obj.data.t_fine, obj.beta_fs);
                if obj.settings.positive
                    obj.data.fitted_fs_fine = max(1e-12, obj.data.fitted_fs_fine);
                end
            end
        end
        
        
        function save_estimates(obj)                                    % Extract results 
            obj.integrate(true);                                            % compute fitted cell trajectories
            obj.data.beta_fs = obj.beta_fs;                                 % store results
            obj.data.beta_fs_history = obj.beta_fs_history;
            obj.data.fitted_fs = obj.fitted_fs;
            obj.data.dfitted_fs = obj.dfitted_fs;
            obj.data.smoothed_fitted_fs = obj.smoothed_fitted;
            obj.data.dsmoothed_fitted_fs = obj.dsmoothed_fitted;
            obj.data.V = obj.V;
            obj.data.varXdX = obj.varXdX;
            obj.data.varbeta = obj.varbeta;
            obj.data.convergence_steps = obj.convergence_steps;
            obj.data.converged = setdiff(1:obj.N, obj.not_converged);
        end

        
        function covariances_partial(obj, rep, converged)               % Parameter uncertainties and residual covariances
            if nargin < 3, converged = false; end                       % for partial observation
            if converged, cells = 1:obj.N; else, cells = obj.not_converged; end

            hidden = setdiff(1:obj.system.K, obj.data.observed);            % hidden/observed state/time indices
            ind_hid = flatten((1:obj.T)'+(obj.T*(hidden-1)));
            ind_obs = flatten((1:obj.T)'+(obj.T*(obj.data.observed-1)));
            
            indices_t = 2:obj.T-1;                                          % time indices without interval ends
            indices_tk = reshape(indices_t' + obj.T*(0:obj.system.K-1), 1, []); % across states
            indices_2tk = reshape(indices_tk' + [0 obj.system.K*obj.T], 1, []); % across state and gradient components
            
            Z = blkdiag(obj.data.basis{:});                                % B-spline basis for all states
            Z_fs = blkdiag(obj.data.basis_gm{:});                          % equivalent with first stage time point grid
            dZ_fs = blkdiag(obj.data.dbasis_gm{:}) / range(obj.data.t);    % corresponding differentiated basis

            if rep > 1                                                      % compute g(.), h(.), and their Jacobians for each cell
                g_all = obj.system.g(obj.smoothed_fitted, obj.data.t);
                dg_all = zeros(obj.system.K, obj.system.P, obj.T, obj.system.K, obj.N);
                dg_all(:, :, :, :, cells) = obj.system.dg(obj.smoothed_fitted(:, :, cells), obj.data.t);
                h_all = obj.system.h(obj.smoothed_fitted, obj.data.t);
                dh_all = obj.system.dh(obj.smoothed_fitted, obj.data.t);
            end
                                                                            % RHS Jacobian for each cell from smoothed measurements
            df_dX_all = obj.system.df(obj.smoothed_fitted, obj.data.t, obj.beta_fs);
            
            beta_permuted = permute(obj.beta_fs, [3 2 1]);                  % vectorized finite difference approximations
            epsilon = max(1e-8, beta_permuted * .001);                      % of solution and RHS parameter sensitivities
            beta_pm_eps = beta_permuted + epsilon .* kron([1; -1], eye(obj.system.P));
            beta_pm_eps = reshape(permute(beta_pm_eps, [2 1 3]), obj.system.P, [])';

                                                                            % compute permuted traces and rhs evaluations
            traces_pm_eps = obj.system.integrate(beta_pm_eps, obj.data, obj.data.t, 1e-4);
            traces_pm_eps = reshape(traces_pm_eps, obj.T, obj.system.K, obj.system.P, 2, obj.N);
            gradients_pm_eps = reshape(obj.system.rhs(reshape(traces_pm_eps, obj.T, obj.system.K, []), obj.data.t, ...
                                                    beta_pm_eps), obj.T, obj.system.K, obj.system.P, 2, obj.N);

            eps_denom = 1./reshape(2*epsilon, 1, 1, obj.system.P, 1, obj.N);% finite difference approximations
            dF_dbeta_all = permute((traces_pm_eps(:, :, :, 1, :) - traces_pm_eps(:, :, :, 2, :)) .* eps_denom, [1:3 5 4]);
            df_dbeta_all = permute((gradients_pm_eps(:, :, :, 1, :) - gradients_pm_eps(:, :, :, 2, :)) .* eps_denom, [1:3 5 4]);
            
            for i = 1
                dF_dbeta = reshape(dF_dbeta_all(:, :, :, i), [], obj.system.P);
                df_dbeta = reshape(df_dbeta_all(:, :, :, i), [], obj.system.P);

                if rep == 1                                                 % smoothing covariance
                    S = diag(max(reshape(obj.data.variances_sm(:, :, i), 1, []), 1e-10));
                    var_delta = tryinv(Z' * (S \ Z));
                    var_XOdXO = [Z_fs; dZ_fs] * var_delta * [Z_fs; dZ_fs]';

                    obj.varbeta(:, :, i) = .25*diag(obj.beta_fs(i, :).^2);% initial (large) parameter uncertainty
                    var_XdXint = [dF_dbeta; df_dbeta] * obj.varbeta(:, :, i) * [dF_dbeta; df_dbeta]';

                    var_XdX = zeros(2 * obj.system.K * obj.T);              % full state/gradient covariance
                    var_XdX(ind_obs + [0 end/2], ind_obs + [0 end/2]) = var_XOdXO;
                    var_XdX(ind_hid + [0 end/2], ind_hid + [0 end/2]) = var_XdXint(ind_hid + [0 end/2], ...
                                                                                       ind_hid + [0 end/2]);
                    
                    obj.varXdX(indices_2tk, indices_2tk, i) = var_XdX(indices_2tk, indices_2tk);

                else
                    dX = flatten(obj.dsmoothed_fitted(indices_t, :, i));    % left-hand side
                    H = flatten(h_all(indices_t, :, i));                    % constant part wrt parameters
                    G = g_all(indices_tk, :, i);                            % linear part wrt parameters
                    
                    Vinv = zeros(obj.system.K * obj.T);                     % inverse residual covariance matrix estimate
                    Vinv(indices_tk, indices_tk) = tryinv(obj.V(indices_tk, indices_tk, i));
                    
                                                                            % construct d[beta]/d[dX] and d[beta]/d[X]
                                                                            % cf. section "S6 First-stage uncertainty estimates for fully observed systems"
                    Th = G' * Vinv(indices_tk, indices_tk) * G;
                    Thinv = tryinv(Th);
                    dbeta_ddX = Thinv * G' * Vinv(indices_tk, indices_tk);  % partial with respect to dX

                    Xi = G' * Vinv(indices_tk, indices_tk) * (dX - H);
                    Thinv_Xi = Thinv * Xi;
                    Pi = zeros(obj.system.P, obj.T, obj.system.K);
                    Psi = zeros(obj.system.P, obj.T, obj.system.K);
                    for k = 1:obj.system.K
                        for j = 2:obj.T-1
                            dg_dxk = dg_all(:, :, j, k, i);
                            Vinv_j = Vinv(j + obj.T*(0:obj.system.K-1), indices_tk);

                            Pi(:, j, k) = (dg_dxk' * Vinv_j * G + G' * Vinv_j' * dg_dxk) * Thinv_Xi;
                            Psi(:, j, k) = dg_dxk' * Vinv_j * (dX-H) - G' * Vinv_j' * dh_all(j + obj.T*(0:obj.system.K-1), k, i);
                        end
                    end                                                     % partial with respect to X
                    dbeta_dX = Thinv * reshape(Psi(:, indices_t, :) - Pi(:, indices_t, :), obj.system.P, []);
                    
                                                                            % delta method approximation
                    obj.varbeta(:, :, i) = [dbeta_dX dbeta_ddX] * obj.varXdX(indices_2tk, indices_2tk, i) * [dbeta_dX dbeta_ddX]';
                    if converged, return, end

                                                                            % iterative update of state/gradient covariance
                    identity_O_indices = zeros(obj.L * obj.T, obj.system.K * obj.T);
                    identity_O_indices(:, ind_obs) = eye(obj.L * obj.T);
                    delta_XdX = zeros(2 * obj.system.K * obj.T);
                    
                    delta_XdX(ind_hid,         indices_2tk) = dF_dbeta(ind_hid, :)  * [dbeta_dX dbeta_ddX];
                    delta_XdX(ind_hid + end/2, indices_2tk) = df_dbeta(ind_hid, :) * [dbeta_dX dbeta_ddX];

                    delta_XdX(ind_obs,         :) = [identity_O_indices   zeros(obj.L * obj.T, obj.system.K * obj.T)];
                    delta_XdX(ind_obs + end/2, :) = [zeros(obj.L * obj.T, obj.system.K * obj.T)   identity_O_indices];
                    
                    obj.varXdX(indices_2tk, indices_2tk, i) = delta_XdX(indices_2tk, indices_2tk) * obj.varXdX(indices_2tk, indices_2tk, i) ...
                                                                                                  * delta_XdX(indices_2tk, indices_2tk)';
                end
                
                df_dX = zeros(obj.system.K * obj.T);                        % residual covariance approximation
                for j = 1:obj.T                                             % restructure RHS Jacobian
                    df_dX(j + obj.T*(0:obj.system.K-1), j + obj.T*(0:obj.system.K-1)) = df_dX_all(j + obj.T*(0:obj.system.K-1), :, i);
                end
                residual_gradient = [-df_dX eye(obj.system.K*obj.T)];       % delta method approximation
                V_unregulated = residual_gradient(indices_tk, indices_2tk) * obj.varXdX(indices_2tk, indices_2tk, i) ...
                                                                           * residual_gradient(indices_tk, indices_2tk)';
                
                regulator = max(1e-12, max(abs(obj.smoothed_fitted(:, :, i)) / 1e6, [], 1));
                regulator = reshape(repmat(regulator, obj.T-2, 1), 1, []);
                                                                            % regulated delta method approximation
                obj.V(indices_tk, indices_tk, i) = V_unregulated + diag(regulator);
            end
        end
    end
end








