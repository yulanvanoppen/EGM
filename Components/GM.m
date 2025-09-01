classdef GM < handle
    properties (SetAccess = private)
        data                                                                % general data and output container
        system                                                              
        settings                                                            % hyperparameters and user input
        
        T                                                                   % number of data time points
        T_gm                                                                % number of first stage time points
        T_fine                                                              % number of time points for smooth plots
        L                                                                   % system dimension
        
        B
        dB
        
        C
        
        smoothed_filtered
        dsmoothed_dfiltered
        
        observed
        hidden

        b
        g
        fdiff
    end
    
    
    methods
        function obj = GM(system, data, settings)                       % Constructor
            obj.data = data;                                                % store data, ODE system, and hyperparameters
            obj.system = system;                                            
            obj.settings = settings;
            [obj.T, obj.L] = size(obj.data.traces);                         % extract trajectory dimensions
            obj.T_gm = length(settings.t_gm);
            
            obj.B = blkdiag(obj.data.basis{:});
            obj.dB = blkdiag(obj.data.dbasis{:});
            
            obj.C = zeros(obj.L, obj.system.K);
            obj.C(:, obj.data.observed) = eye(obj.L);
            
            obj.smoothed_filtered = zeros(obj.T, obj.system.K);
            obj.dsmoothed_dfiltered = zeros(obj.T, obj.system.K);
            obj.smoothed_filtered(:, data.observed) = data.smoothed;
            obj.dsmoothed_dfiltered(:, data.observed) = data.dsmoothed;
            
            obj.observed = data.observed;
            obj.hidden = 1:obj.system.K;
            obj.hidden = obj.hidden(~ismember(1:obj.system.K, obj.observed));

            order = 3; framelen = 7;
            [obj.b, obj.g] = sgolay(order, framelen);

            [~, g5] = sgolay(order, 5);
            [~, g3] = sgolay(order-2, 3);

            t = data.t;
            dt = [t(2)-t(1), (t(3:end) - t(1:end-2))/2, t(end)-t(end-1)];
            diagonals = repmat(obj.g(:, 2)', obj.T, 1);
            obj.fdiff = spdiags(diagonals, -(framelen-1)/2:(framelen-1)/2, obj.T, obj.T);

            obj.fdiff([1:3 end-2:end], :) = 0;

            obj.fdiff(3, 1:5) = g5(:, 2)';
            obj.fdiff(end-2, end-4:end) = g5(:, 2)';
            
            obj.fdiff(2, 1:3) = g3(:, 2)';
            obj.fdiff(end-1, end-2:end) = g3(:, 2)';

            obj.fdiff(1, 1:2) = [-1 1];
            obj.fdiff(end, end-1:end) = [-1 1];

            obj.fdiff = obj.fdiff ./ dt;

            % dt = [t(2)-t(1), t(3:end) - t(1:end-2), t(end)-t(end-1)];
            % obj.fdiff = spdiags(repmat([-1 0 1], obj.T, 1), -1:1, obj.T, obj.T);
            % obj.fdiff([1 end]) = [-1 1];
            % obj.fdiff = obj.fdiff ./ dt';
        end
        
        
        function update_settings(obj, varargin)
            if isstruct(varargin{1})
                obj.settings = varargin{1};
            else
                obj.settings(varargin{1}) = varargin{2};
            end
        end
        
        
        function beta = estimate(obj, beta, filtered)
            t = obj.data.t;

            % filtered_dxbar = obj.system.rhs(filtered.xbar', t, beta);

            % dt = [t(2)-t(1), t(3:end) - t(1:end-2), t(end)-t(end-1)];
            % fdiff = diag(ones(1, obj.T_gm-1), 1) - diag(ones(1, obj.T_gm-1), -1);
            % fdiff([1 end]) = [-1 1];
            % fdiff = fdiff ./ dt';
            % filtered_dxbar = fdiff * filtered.xbar';

            filtered_dxbar = obj.fdiff * filtered.xbar';

            obj.smoothed_filtered(:, obj.hidden) = filtered.xbar(obj.hidden, :)';
            obj.dsmoothed_dfiltered(:, obj.hidden) = filtered_dxbar(:, obj.hidden);

            G_smoothed_filtered = obj.system.g(obj.smoothed_filtered(2:end-1, :), t(2:end-1));
            H_smoothed_filtered = obj.system.h(obj.smoothed_filtered(2:end-1, :), t(2:end-1));

            design = G_smoothed_filtered;
            const = H_smoothed_filtered;
            response = obj.dsmoothed_dfiltered(2:end-1, :) - const;

            V = obj.covariances(beta, filtered);

            beta = Optimization.QPGLS(design, response, nearestSPD(V))';
        end
        
        
        function variances = covariances(obj, beta, filtered)
            t = obj.data.t;
            K = obj.system.K;
            KT = K*obj.T;
            LT = obj.L*obj.T;
            M = size(obj.B, 2);
            
            ind_hid = flatten((1:obj.T)'+(obj.T*(obj.hidden-1)));           % hidden/observed state/time indices
            ind_obs = flatten((1:obj.T)'+(obj.T*(obj.observed-1)));

            indices_t = 2:obj.T-1;                                          % time indices without interval ends
            indices_tk = reshape(indices_t' + obj.T*(0:obj.system.K-1), 1, []); % across states
            indices_2tk = reshape(indices_tk' + [0 obj.system.K*obj.T], 1, []); % across state and gradient components

            % dRHS_filtered = obj.system.df(filtered.xbar', obj.data.t, beta);
            dRHS_mixed = obj.system.df(obj.smoothed_filtered, obj.data.t, beta);

            % dRHS = zeros(KT);
            % for j = 1:obj.T
            %     dRHS(j + obj.T*(0:K-1), j + obj.T*(0:K-1)) = dRHS_filtered(j + obj.T*(0:K-1), :);
            % end
            % delta_mult = [eye(KT)       zeros(KT, M);
            %               dRHS          zeros(KT, M);
            %               zeros(M, KT)  eye(M)];


            % dt = [t(2)-t(1), t(3:end) - t(1:end-2), t(end)-t(end-1)];
            % fdiff = diag(ones(1, obj.T_gm-1), 1) - diag(ones(1, obj.T_gm-1), -1);
            % fdiff([1 end]) = [-1 1];
            % fdiff = fdiff ./ dt';
            % fdiff_cell = cell(1, K);
            % [fdiff_cell{:}] = deal(fdiff);
            % fdiff_full = blkdiag(fdiff_cell{:});

            fdiff_cell = cell(1, K);
            [fdiff_cell{:}] = deal(obj.fdiff);
            fdiff_full = blkdiag(fdiff_cell{:});


            delta_X_dX_y = [eye(KT)       zeros(KT, LT);
                            fdiff_full    zeros(KT, LT);
                            zeros(LT, KT)  eye(LT)];


           
            covar = delta_X_dX_y * filtered.covar * delta_X_dX_y';


            lincomb_data = blkdiag(obj.data.lincomb_data{:});
            
            delta_X_dX = zeros(2*KT, 2*KT+LT);
            delta_X_dX(ind_hid, ind_hid) = eye(KT-LT);
            delta_X_dX(ind_obs, end-LT+1:end) = obj.B * lincomb_data;
            delta_X_dX(ind_hid+KT, ind_hid+KT) = eye(KT-LT);
            delta_X_dX(ind_obs+KT, end-LT+1:end) = obj.dB * lincomb_data;

            variances_XdX = delta_X_dX * covar * delta_X_dX';

            dRHS = zeros(KT);                                               % variance of X, dX, AND dX - G(X)beta - H(X)
            for j = 1:obj.T
                dRHS(j + obj.T*(0:K-1), j + obj.T*(0:K-1)) = dRHS_mixed(j + obj.T*(0:K-1), :);
            end
            delta_residuals = [-dRHS eye(KT)];

            V = delta_residuals(indices_tk, indices_2tk) * variances_XdX(indices_2tk, indices_2tk) ...
                                                         * delta_residuals(indices_tk, indices_2tk)';

            regulator = max(1e-12, max(abs(obj.smoothed_filtered(:, :)) / 1e6, [], 1));
            regulator = reshape(repmat(regulator, obj.T-2, 1), 1, []);
            variances = nearestSPD(V + diag(regulator));
        end
    end
end