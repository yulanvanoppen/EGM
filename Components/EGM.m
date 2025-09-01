classdef EGM < handle
    
    properties (SetAccess = private)
        data                                                                % measurements struct
        system                                                              % ODE system object
        
        autoknots                                                           % logical for placement heuristoic
        knots                                                               % B-spline knots for each state
        interactive                                                         % logical for interactive smoothing app use
        
        lb                                                                  % parameter space lower bounds
        ub                                                                  % parameter space upper bounds
        positive                                                            % logical for forced state positivity
        t_gm                                                                % first stage optimization grid (GMGTS)
        
        niterSM                                                             % #iterations for smoothing
        tolSM                                                               % convergence tolerance for smoothing
        niterGM                                                             % #iterations for the first stage
        tolGM                                                               % convergence tolerance for the first stage

        init
        P_CV
        sigma
        Q_mult

        
        settings_smoothing                                                  % structs containing default/user preferences
        settings_filter
        settings_inference
        output                                                              % inference output struct
        
        smoother                                                            % internal smoother object
        filter                                                              % internal EKF object
        gm                                                                  % internal gradient matching object
        
        beta
        filtered
    end
    
    
    methods
        %% Constructor -----------------------------------------------------
        function obj = EGM(system, data, varargin)
            [system, data] = obj.parse_initial(system, data, varargin{:});
            
            default_t = 0:size(data, 1)-1;
            default_observed = 1:size(data, 2);
            
            initial = system.k0';
            default_AutoKnots = true;
            default_Knots = repmat({linspace(data.t(1), data.t(end), round((data.T-1)/2)+1)}, ...
                                   1, length(data.observed));
            default_InteractiveSmoothing = false;
            
            default_LB = .25 * initial;
            default_UB = 4 .* initial + .0001 * mean(initial);
            default_PositiveStates = true;
            % default_TimePoints = data.t(1) + (0:.1:1) * range(data.t);
            default_TimePoints = data.t;
            
            default_MaxIterationsSM = 20;
            default_ConvergenceTolSM = 1e-3;
            default_MaxIterationsGM = 50;
            default_ConvergenceTolGM = 2e-3;

            default_InitialConditions = system.x0';
            default_PCV = .5;
            default_Sigma = .1;
            default_QMult = 1;
            
            parser = inputParser;
            parser.KeepUnmatched = true;
            addRequired(parser, 'system', @(x) isa(x, 'System') || isstring(string(x)) && numel(string(x)) == 1);
            addRequired(parser, 'data', @(x) isstruct(x) || isnumeric(x) && ndims(x) == 3 && size(x, 1) > 1);
            addOptional(parser, 't', default_t, @(x) isnumeric(x) && numel(unique(x)) == data.T);
            addOptional(parser, 'observed', default_observed, @(x) isnumeric(x) && numel(unique(x)) == data.L);
            
            addParameter(parser, 'AutoKnots', default_AutoKnots, @islogical);
            addParameter(parser, 'Knots', default_Knots, @(x) (iscell(x) && length(x) == data.L ...
                                                               && all(cellfun(@isnumeric, x))) ...
                                                           || isnumeric(x));
            addParameter(parser, 'InteractiveSmoothing', default_InteractiveSmoothing, @islogical);

            addParameter(parser, 'LB', default_LB, @(x) all(x < initial));
            addParameter(parser, 'UB', default_UB, @(x) all(x > initial));
            addParameter(parser, 'PositiveStates', default_PositiveStates, @islogical);
            addParameter(parser, 'TimePoints', default_TimePoints, @(x) all(data.t(1) <= x & x <= data.t(end)));
            
            addParameter(parser, 'MaxIterationsSM', default_MaxIterationsSM, @isscalar);
            addParameter(parser, 'ConvergenceTolSM', default_ConvergenceTolSM, @isscalar);
            addParameter(parser, 'MaxIterationsGM', default_MaxIterationsGM, @isscalar);
            addParameter(parser, 'ConvergenceTolGM', default_ConvergenceTolGM, @isscalar);

            addParameter(parser, 'InitialConditions', default_InitialConditions, @(x) numel(x) == system.K);
            addParameter(parser, 'PCV', default_PCV, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'Sigma', default_Sigma, @(x) isscalar(x) && x > 0);
            addParameter(parser, 'QMult', default_QMult, @(x) isscalar(x) && x > 0);
            
            parse(parser, system, data, varargin{:});
            [obj.system, obj.data] = obj.parse_initial(parser.Results.system, parser.Results.data, varargin{:});
            obj.parse_parameters(parser);
            
            obj.data.T_fine = 81;
            obj.data.t_fine = linspace(obj.data.t(1), obj.data.t(end), obj.data.T_fine);
            
            weights = ones(length(obj.t_gm), obj.system.K);                 % omit interval ends
            weights([1 end], :) = 0;
            
            obj.settings_smoothing = struct('order', 4, 'autoknots', obj.autoknots, 'knots', {obj.knots}, ...
                                            'positive', obj.positive, 't_gm', obj.t_gm, 'niter', obj.niterSM, ...
                                            'tol', obj.tolSM, 'interactive', obj.interactive);
            obj.settings_filter = struct('sigma', obj.sigma, 'Q_mult', obj.Q_mult, 'P_CV', obj.P_CV, 'x_init', obj.init, 't_gm', obj.t_gm);
            obj.settings_inference = struct('lb', obj.lb, 'ub', obj.ub, 'positive', obj.positive, 't_gm', obj.t_gm, ...
                                            'weights', weights, 'niter', obj.niterGM, 'tol', obj.tolGM);
        end

                                                                        % Basic parsing to allow conditional input validations
        function [system, data] = parse_initial(~, system, data, varargin)  
            if ~isa(system, 'System')                                       % process model file if provided
                namevalue = arrayfun(@(idx) iscellstr(varargin(idx)) || isstring(varargin{idx}), 1:length(varargin));
                first_namevalue = find(namevalue, 1);
                if isempty(first_namevalue), first_namevalue = length(varargin)+1; end
                system = System(string(system), varargin{first_namevalue:end});
            end
            if isstruct(data)                                               % default any missing fields
                if ~isfield(data, 'traces') && isfield(data, 'y'), data.traces = data.y; end
                if ~isfield(data, 't'), data.t = 0:size(data.traces, 1)-1; end
                if ~isfield(data, 'observed'), data.observed = 1:size(data.traces, 2); end
                if ~isfield(data, 'init'), data.init = system.x0' + 1e-4; end
            else                                                            % components provided separately
                traces = data;                                              % array with measurements instead of struct
                if ~iscellstr(varargin(1)) && ~isstring(varargin{1})        % recursively check if optional or Name/Value
                    t = sort(unique(reshape(varargin{1}, 1, [])));
                    if ~iscellstr(varargin(2)) && ~isstring(varargin{2})
                        observed = sort(unique(reshape(varargin{2}, 1, [])));
                    else
                        observed = 1:size(traces, 2);
                    end
                else
                    t = 0:size(traces, 1)-1;
                    observed = 1:size(traces, 2);
                end                                                         % compile into struct
                data = struct('traces', traces, 't', t, 'observed', observed, 'init', init);
            end
            [data.T, data.L, data.N] = size(data.traces);                   % include dimensions for notational convenience
        end


        function parse_parameters(obj, parser)                          % Parse Name/Value constructor arguments
            obj.autoknots = parser.Results.AutoKnots;
            obj.knots = cell(1, obj.data.L);
            parsed_knots = parser.Results.Knots;
            if ~iscell(parsed_knots)
                parsed_knots = repmat({parsed_knots}, 1, obj.data.L);
            end
            for state = 1:length(obj.data.observed)
                truncated = max(obj.data.t(1), min(obj.data.t(end), parsed_knots{state}));
                arranged = sort(unique([obj.data.t(1) reshape(truncated, 1, []) obj.data.t(end)]));
                obj.knots{state} = (arranged - obj.data.t(1)) / range(obj.data.t);
            end
            obj.autoknots = parser.Results.AutoKnots && ismember("Knots", string(parser.UsingDefaults));
            obj.interactive = parser.Results.InteractiveSmoothing;
            
            obj.lb = parser.Results.LB;
            obj.ub = parser.Results.UB;
            obj.positive = parser.Results.PositiveStates;
            obj.t_gm = sort(unique(parser.Results.TimePoints));

            obj.niterSM = max(1, round(parser.Results.MaxIterationsSM));
            obj.tolSM = max(1e-12, parser.Results.ConvergenceTolSM);
            obj.niterGM = max(1, round(parser.Results.MaxIterationsGM));
            obj.tolGM = max(1e-12, parser.Results.ConvergenceTolGM);

            obj.init = reshape(parser.Results.InitialConditions, 1, []);
            obj.P_CV = parser.Results.PCV;
            obj.sigma = parser.Results.Sigma;
            obj.Q_mult = parser.Results.QMult;
        end
        
        
        %% Estimation ------------------------------------------------------
        function out = estimate(obj, beta_init, ~)
            ws = warning('error', 'MATLAB:nearlySingularMatrix');
            silent = nargin == 3;
            
            obj.smoother = Smoother(obj.system, obj.data, obj.settings_smoothing);
            
            tic
            obj.output = obj.smoother.smooth();
            
            toc_sm = obj.output.toc_sm;
            if ~silent, toc_sm, end     %#ok<NOPRT>
            
            obj.filter = Filter(obj.system, obj.output, obj.settings_filter);
            obj.gm = GM(obj.system, obj.output, obj.settings_inference);
            % obj.gm = EMGM(obj.system, obj.output, obj.settings_inference);
            
            obj.beta = zeros(obj.settings_inference.niter+1, obj.system.P);
            obj.beta(1, :) = beta_init;
            
            tic
            for iter = 1:obj.settings_inference.niter
                fprintf('%d ', iter)
                
                obj.filtered = obj.filter.EKF(obj.beta(iter, :));            % filter and gradient matching
                obj.beta(iter+1, :) = obj.gm.estimate(obj.beta(iter, :), obj.filtered);
                % obj.beta(rep+1, :) = obj.beta(rep, :);
                
                if ~mod(iter, 10), fprintf('(%.3e)\n', eucl_rel(obj.beta(iter, :), obj.beta(iter+1, :))), end
                
                if iter == obj.settings_inference.niter || ...
                   eucl_rel(obj.beta(iter, :), obj.beta(iter+1, :)) < obj.settings_inference.tol
                    obj.beta = obj.beta(1:iter+1, :);
                    fprintf('(%.3e)\n', eucl_rel(obj.beta(iter, :), obj.beta(iter+1, :)))
                    % % % % % % % obj.filtered = obj.filter.EKF(obj.beta(end, :));
                    % % % % % % % obj.gm.estimate(obj.beta(rep, :), obj.filtered);
                    break
                end
            end
            fprintf('\n')
            
            if ~silent, toc_est = toc, else, toc_est = toc; end     %#ok<NOPRT>
            
            
            if ~silent
                fprintf('total time: %.3f seconds\n', toc_sm + toc_est)
            end


            
            fitted = obj.system.integrate(obj.beta(end, :), obj.data);   % integrate system and compute rhs
            fitted_fine = obj.system.integrate(obj.beta(end, :), obj.data, obj.data.t_fine);

            dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta(end, :));
            dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta(end, :));

            if obj.positive
                fitted = max(1e-12, fitted);                          % force positive
                fitted_fine = max(1e-12, fitted_fine);
            end
            
            obj.output.fitted = fitted;
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;


            
            obj.output.time = [toc_sm toc_est];
            obj.output.beta = obj.beta(end, :);
            obj.output.filtered = obj.filtered;
            out = obj.output;
            
            warning(ws);
        end


        function out = estimate_iterative(obj, beta_init, x_init)       % Estimate using GMGTS iterative algorithm
            ws = warning('error', 'MATLAB:nearlySingularMatrix');
            silent = nargin == 3;
            
            obj.smoother = Smoother(obj.system, obj.data, obj.settings_smoothing);
            
            tic
            obj.output = obj.smoother.smooth();
            
            toc_sm = obj.output.toc_sm;
            if ~silent, toc_sm, end     %#ok<NOPRT>
            
            obj.gm = FSGMGTS(obj.output, obj.system, obj.settings_inference, beta_init, x_init);
            
            tic
            out = obj.gm.optimize();
            toc_est = toc

            obj.beta = out.beta_fs;

            fitted = obj.system.integrate(obj.beta, obj.data);   % integrate system and compute rhs
            fitted_fine = obj.system.integrate(obj.beta, obj.data, obj.data.t_fine);

            dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
            dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);

            if obj.positive
                fitted = max(1e-12, fitted);                          % force positive
                fitted_fine = max(1e-12, fitted_fine);
            end
            
            obj.output.fitted = fitted;
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;

            obj.output.time = [toc_sm toc_est];
            obj.output.beta = obj.beta;
            out = obj.output;
            
            warning(ws);
        end


        function out = estimate_TM(obj, beta_init, x_init)       % Estimate using GMGTS iterative algorithm
            ws = warning('error', 'MATLAB:nearlySingularMatrix');
            silent = nargin == 3;
            
            tm = FSGTS(obj.data, obj.system, obj.settings_inference, beta_init, x_init);
            
            tic
            out = tm.optimize(true);
            if silent, toc_est = toc; else, toc_est = toc, end

            obj.beta = out.beta_fs;
            obj.output.init = out.init;

            fitted = obj.system.integrate(obj.beta, obj.data);   % integrate system and compute rhs
            fitted_fine = obj.system.integrate(obj.beta, obj.data, obj.data.t_fine);

            dfitted = obj.system.rhs(fitted, obj.data.t, obj.beta);
            dfitted_fine = obj.system.rhs(fitted_fine, obj.data.t_fine, obj.beta);

            if obj.positive
                fitted = max(1e-12, fitted);                          % force positive
                fitted_fine = max(1e-12, fitted_fine);
            end
            
            obj.output.fitted = fitted;
            obj.output.dfitted = dfitted;
            obj.output.fitted_fine = fitted_fine;
            obj.output.dfitted_fine = dfitted_fine;

            obj.output.time = [0 toc_est];
            obj.output.beta = obj.beta;
            out = obj.output;
            
            warning(ws);
        end
    end
end