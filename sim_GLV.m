% SWITCH TO LINEAR SYSTEM (HARMONIC OSCILLATOR)
% IMPLEMENT EKF WITH COVARIANCE
% CHECK AGAINST BOOTSTRAP + VARYING SPLINE UNCERTAINTY AND CORRELATION


% EXTEND TEST
% EXPECTED SUMS OF SQUARES
% IMPLEMENT CLASS


%% Setup system ------------------------------------------------------------

set(groot,'defaultAxesTickLabelInterpreter','tex');  
set(groot,'defaulttextinterpreter','tex');
set(groot,'defaultLegendInterpreter','tex');

% clearvars
close all
rng(0)

% system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters = ["r"]);
% save('system_GLV.mat', 'system')
load('system_GLV.mat')

 
% Generate data -----------------------------------------------------------
generator = Generator(system, N=1, t=[0:.5:100], error_const=eps, error_std=.05, D_mult=eps, observed=system.K);
[data, ground_truth] = generator.generate();
plot(generator)


%% Inference
% beta0 = [.5 .1 .8 .2];
beta0 = [.5 .5];
% beta0 = [.8 .2];

estimator = GMEKF(system, data, Knots=[1:9], MaxIterationsGM=20, PCV = .25, InitialConditions=[.1 .1 .1], ...
                  ConvergenceTolGM=1e-6, Sigma=1E-2, PositiveStates=false);
out = estimator.estimate(beta0)



%% Plot smoothing
col = parula(system.K+2);
col = col(2:end-1, :);

scales = ones(size(mean(ground_truth.original)));

figure(2)
h1 = plot(data.t, data.traces ./ scales(:, data.observed), '.-');
set(h1, {'color'}, num2cell(col(data.observed, :), 2));

hold on
h2 = plot(data.t, ground_truth.original ./ scales, '-');
set(h2, {'color'}, num2cell(col, 2));

h3 = plot(out.t_fine, out.smoothed_fine ./ scales(:, data.observed), '--');
set(h3, {'color'}, num2cell(col(data.observed, :), 2));

hold off

title('Smoothing')
legend([h1(1), h2(2), h3(1)], 'Data', 'Ground truth', 'Smoothed measurements', Location='northwest')
xlabel('Time (min)')
ylabel('Value (A.U.)')


%% Plot filtering
col = parula(system.K+2);
col = col(2:end-1, :);

scales = ones(size(mean(ground_truth.original)))

f = figure(3);
h1 = plot(data.t, data.traces ./ scales(:, data.observed), 'o');
set(h1, {'color'}, num2cell(col(data.observed, :), 2));

hold on
h2 = plot(data.t, ground_truth.original ./ scales, '-');
set(h2, {'color'}, num2cell(col, 2));

h4 = plot(data.t, out.filtered.xbar ./ scales', '-.', LineWidth=1.5);
set(h4, {'color'}, num2cell(col, 2));

h5 = plot(data.t, [out.filtered.xbar+out.filtered.sd;
                   out.filtered.xbar-out.filtered.sd] ./ [scales scales]', '-.');
set(h5, {'color'}, num2cell(col([1:end 1:end], :), 2));

% h6 = plot(out.t_fine, out.fitted_fine ./ scales, ':', LineWidth=2);
% set(h6, {'color'}, num2cell(col, 2));

hold off

% title('Filtering')
% legend([h1(1), h2(2), h4(1), h6(1)], 'Data', 'Ground truth', 'Filtered distribution', 'Predicted dynamics', Location='northwest')
legend([h1(1), h2(2), h4(2)], 'Data', 'Ground truth', 'Filtered distribution', Location='northeast')
xlabel('Time (min)')
ylabel('Value (A.U.)')

f.Units = 'centimeters';        % set figure units to cm
f.PaperUnits = 'centimeters';   % set pdf printing paper units to cm
f.PaperSize = f.Position(3:4);  % assign to the pdf printing paper the size of the figure
print -dpdf ../figures/oscillator_filtered2;           % print the figure

%% Plot gradients
% col = parula(8);
% col = col(2:7, :);
% 
% Xhat = estimator.gm.smoothed_filtered;
% dXhat = estimator.gm.dsmoothed_dfiltered;
% 
% G_smoothed_filtered = system.g(Xhat, data.t);
% H_smoothed_filtered = system.h(Xhat, data.t);
% RHS = system.rhs(Xhat, data.t, out.beta);
% 
% design = G_smoothed_filtered;
% const = H_smoothed_filtered;
% response = dXhat - const;
% 
% 
% 
% figure(4)
% tiledlayout(1, 2)
% for k = 1:2
%     nexttile(k)
%     plot(data.t, ground_truth.doriginal(:, k), 'o', Color=col(k, :));
%     hold on
%     plot(data.t, dXhat(:, k), '--', LineWidth=1.5, Color=col(k, :));
%     plot(data.t, RHS(:, k), '-.', LineWidth=1.5, Color=col(k, :));
%     hold off
%     title(system.states(k))
% end
% 
% % h5 = plot(data.t, [out.filtered.xbar+out.filtered.sd;
% %                    out.filtered.xbar-out.filtered.sd] ./ [scales scales]', '-.');
% % set(h5, {'color'}, num2cell(col([1:end 1:end], :), 2));


%% Bootstrap covariance
% M = 100;
% 
% xbar_estimate = out.filtered.xbar;
% covar_estimate = out.filtered.covar;
% 
% xbar_y_bootstrap = zeros([numel(xbar_estimate)+numel(data.traces) M]);
% 
% for m = 1:M
%     [data, ground_truth] = generator.generate();
% 
%     beta0 = [.8 .2];
% 
%     estimator = GMEKF(system, data, Knots=[0 3 75], MaxIterationsGM=1, ...
%                       ConvergenceTolGM=1e-2, Sigma=.01);
%     out = estimator.estimate(beta0);
% 
%     xbar_bootstrap(:, m) = [reshape(out.filtered.xbar', 1, []) reshape(data.traces, 1, [])];
% end
% 
% covar_bootstrap = cov(xbar_bootstrap');

%% Bootstrap
% rng(0);
% 
% nrep = 100;
% beta_true = ground_truth.beta(1, :);
% beta0_ODE = beta_true .* (1+(rand(nrep, 2)-.5)*1.5);
% beta0_filtered = beta_true .* (1+(rand(nrep, 2)-.5)*1.5);
% betaN_ODE = zeros(size(beta0_ODE));
% betaN_filtered = zeros(size(beta0_ODE));
% 
% dist_init_ODE = zeros(nrep, 1);
% dist_init_filtered = zeros(nrep, 1);
% dist_ODE = zeros(nrep, 1);
% dist_filtered = zeros(nrep, 1);
% 
% for rep = 1:nrep
%     if ~mod(rep, 1), fprintf('%d ', rep); end
%     if ~mod(rep, 10), fprintf('\n'); end
%     beta = GMEKF1(beta0_ODE(rep, :), system, B, dB, data_EKF, estimator.results_GMGTS);
%     betaN_ODE(rep, :) = beta(:, end)';
%     
%     dist_init_ODE(rep) = norm(beta_true - beta0_ODE(rep, :));
%     dist_ODE(rep) = norm(beta_true - betaN_ODE(rep, :));
% end
% 
% for rep = 1:nrep
%     if ~mod(rep, 1), fprintf('%d ', rep); end
%     if ~mod(rep, 10), fprintf('\n'); end
%     beta = GMEKF1(beta0_ODE(rep, :), system, B, dB, data_EKF, estimator.results_GMGTS, 40);
%     betaN_filtered(rep, :) = beta(:, end)';
%     
%     dist_init_filtered(rep) = norm(beta_true - beta0_filtered(rep, :));
%     dist_filtered(rep) = norm(beta_true - betaN_filtered(rep, :));
% end
% 
% figure(6)
% hold off
% scatter(dist_init_ODE, dist_ODE, 50, 'filled', 'MarkerFaceAlpha', .2, 'MarkerEdgeAlpha', .2)
% hold on
% scatter(dist_init_filtered, dist_filtered, 50, 'filled', 'MarkerFaceAlpha', .2, 'MarkerEdgeAlpha', .2)
% legend('20 iterations', '40 iterations', 'Location', 'northwest')
% xlabel('initial absolute error')
% ylabel('final absolute error')
% ylim([0, .12])


