classdef Filter < handle
    properties (SetAccess = private)
        data                                                                % general data and output container
        system                                                              
        settings                                                            % hyperparameters and user input
        
        T                                                                   % number of data time points
        T_gm                                                                % number of first stage time points
        T_fine                                                              % number of time points for smooth plots
        L                                                                   % system dimension
        N                                                                   % number of cells
        
        sigma
        S
        Om
        
        x_init
        P_init
        
        B
        dB
        
        C
    end
    
    
    methods
        function obj = Filter(system, data, settings)                   % Constructor
            obj.data = data;                                                % store data, ODE system, and hyperparameters
            obj.system = system;                                            
            obj.settings = settings;
            [obj.T, obj.L, obj.N] = size(obj.data.traces);                  % extract trajectory dimensions
            obj.T_gm = length(settings.t_gm);
            
            obj.sigma = obj.settings.sigma;
            obj.S = @(x) obj.sigma .* ones(size(x));
            obj.Om = blkdiag(data.var_delta{:});
            
            obj.x_init = obj.settings.x_init;
            obj.P_init = diag((obj.settings.P_CV .* obj.x_init).^2);
            
            obj.B = blkdiag(obj.data.basis{:});
            obj.dB = blkdiag(obj.data.dbasis{:});
            
            obj.C = zeros(obj.L, obj.system.K);
            obj.C(:, obj.data.observed) = eye(obj.L);


            obj.data.variances_sm = obj.data.variances_sm;
        end
        
        function update_settings(obj, varargin)
            if isstruct(varargin{1})
                obj.settings = varargin{1};
            else
                obj.settings(varargin{1}) = varargin{2};
            end
        end

        
        function filtered = EKF(obj, beta)
            tic
            K = obj.system.K;
            KT = K * obj.T;
            y = obj.data.traces';
            
            xbar = zeros(K, obj.T);                                         % allocate
            P = zeros(K, K, obj.T);
            sd = zeros(K, obj.T);

            xbar(:, 1) = obj.x_init;                                        % initialize
            P(:, :, 1) = obj.P_init;
            sd(:, 1) = sqrt(diag(P(:, :, 1)));
            xbar_pre = xbar;
            P_pre = P;
                                                                            % Process first data point
            [KG, xbar(:, 1), P(:, :, 1), sd(:, 1)] = obj.Kalman_gain(1, y, xbar_pre, P_pre);
            covar = obj.update_covariance(1, KG, P(:, :, 1));
            
            for it = 2:obj.T                                                % solve original and linear matrix ODEs
                [tout, out, dFdx0] = obj.integrate_augmented(beta, xbar(:, it-1), ...
                                                             [obj.data.t_data(it-1) obj.data.t_data(it)]);
                                                       
                [Q_it, Phi_it] = obj.cumulative_noise(tout, out);           % approximate cumulative noise integral
                
                xbar_pre(:, it) = out(end, 1:K);                            % update prior estimates at current time point
                P_pre(:, :, it) = Phi_it * P(:, :, it-1) * Phi_it' + Q_it;
                                                                            % apply Kalman gain and expand covariance matrix
                [KG, xbar(:, it), P(:, :, it), sd(:, it)] = obj.Kalman_gain(it, y, xbar_pre, P_pre);
                covar = obj.update_covariance(it, KG, P(:, :, it), covar, dFdx0);
            end
            
            reordered_K = reshape((1:K) + K*(0:obj.T-1)', 1, []);           % group time points instead of states
            reordered_L = reshape((1:obj.L) + obj.L*(0:obj.T-1)', 1, []);
            covar = covar([reordered_K KT+reordered_L], [reordered_K KT+reordered_L]);
            
                                                                            % save results to output struct
            filtered = struct('xbar', xbar, 'P', P, 'sd', sd, 'covar', covar);
        end
        
                                                                        % Compute and apply Kalman gain
        function [KG, xbar, P, sd] = Kalman_gain(obj, it, y, xbar_pre, P_pre)
            K = obj.system.K;
            I = eye(K);
            
            y = y(:, it);
            xbar_pre = xbar_pre(:, it);
            P_pre = P_pre(:, :, it);
            Sigma = diag(obj.data.variances_sm(it, :));
            
            KG = P_pre * obj.C' / (obj.C * P_pre * obj.C' + Sigma);
            
            xbar = xbar_pre(:, 1) + KG * (y - obj.C * xbar_pre);
            
            P = (I - KG*obj.C) * P_pre * (I - KG*obj.C)' + KG * Sigma * KG';
            sd = sqrt(diag(P));
        end
        
        
        function covar = update_covariance(obj, it, KG, P, covar, dFdx0)  % Expand covar matrix
            K = obj.system.K;
            TK = K * obj.T;
            
            if it == 1
                Sigma = diag(obj.data.variances_sm(1, :));
                covar = zeros(obj.T*(K+obj.L));
                covar(1:K, 1:K) = P;
                covar(1:K, TK+1:TK+obj.L) = KG * Sigma;
                covar(TK+(1:obj.L), 1:K) = covar(1:K, TK+(1:obj.L))';
                covar(TK+(1:obj.L), TK+(1:obj.L)) = Sigma;
            else
                Sigma = diag(obj.data.variances_sm(it, :));
                I = eye(K);
                
                factor = (I - KG*obj.C) * dFdx0;
                
                U = covar(1:K*(it-1), K*(it-2) + (1:K));
                covar(1:K*(it-1), K*(it-1) + (1:K)) = U * factor';          % update Uin, i=1,...,n-1

                covar(K*(it-1) + (1:K), K*(it-1) + (1:K)) = P;              % update Unn

                W = covar(K*(it-2) + (1:K), obj.T*K + (1:obj.L*(it-1)));    % update Win, i=1,...,n-1
                covar(K*(it-1) + (1:K), obj.T*K + (1:obj.L*(it-1))) = factor * W;

                                                                            % update Wnn
                covar(K*(it-1) + (1:K), obj.T*K + obj.L*(it-1) + (1:obj.L)) = KG * Sigma;
                
                covar(obj.T*K + obj.L*(it-1) + (1:obj.L), ...               % add measurement variance
                      obj.T*K + obj.L*(it-1) + (1:obj.L)) = Sigma;
                
                covar = triu(covar) + triu(covar, 1)';                      % symmetrize
            end
        end
        
                                                                        % Integrate auxiliary IQM tools model
        function [tout, yout, dFdx0] = integrate_augmented(obj, beta, x0, trange)
            tout = linspace(trange(1), trange(2), 101);
            yout = obj.system.auxiliary(beta, x0', tout);

            dFdx0 = zeros(length(x0));

            x0_permuted = x0 + 1e-8 .* kron([1 -1], eye(length(x0)));
            for k = 1:length(x0)
                yout_plus_eps = obj.system.integrate(beta, struct('init', x0_permuted(:, k)'), trange, 1e-4);
                yout_minus_eps = obj.system.integrate(beta, struct('init', x0_permuted(:, k+end/2)'), trange, 1e-4);
                dFdx0(:, k) = (yout_plus_eps(end, :) - yout_minus_eps(end, :)) / 2e-8;
            end
        end
        
        
        function [Q_it, Phi_it] = cumulative_noise(obj, tout, out)      % Integrate linear matrix ODE and cumulative noise
            K = obj.system.K;
            
            Phi = permute(reshape(out(:, K+1:end), [], K, K), [2 3 1]);

            S_eval = obj.S(out(:, 1:K)');

            integrand = zeros(size(Phi));
            for tidx = 1:length(tout)
                Phiti = Phi(:, :, tidx);
                Sti = diag(S_eval(:, tidx));
                integrand(:, :, tidx) = Phiti * (Sti * Sti') * Phiti';
            end

            Q_it = reshape(trapz(tout, integrand, 3), K, K);
            Phi_it = reshape(Phi(:, :, end), K, K);
        end
        
    end
end