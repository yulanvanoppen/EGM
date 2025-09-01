%% Setup system ------------------------------------------------------------
clearvars
close all

% system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters = ["r"]);
% save('system_GLV.mat', 'system')
load('system_GLV.mat')


%% Generate data -----------------------------------------------------------
observed = 2:system.K;                                                      % uncomment if L=2
% observed = system.K;                                                      % uncomment if L=1

dt = .5;
if length(observed)==2, noise = 0.1; else, noise = 0.05; end
init_error = 5.^(-1:.5:1) - 1;

generator = Generator(system, N=1, t=0:dt:10, error_std=noise, ...
                      error_const=.01, D_mult=eps, observed=observed);
[data, ground_truth] = generator.generate();


%% Assess convergence ------------------------------------------------------
magnitudes = linspace(0, 2, 21);
converged_EGM = 999 * ones(21, 21, 4);
converged_TM = 999 * ones(21, 21, 4);
converged_GM = 999 * ones(21, 21, 4);

for init_idx = 1:4
for b1_idx = 1:21
for b2_idx = 1:21
    error = init_error(init_idx);   
    beta0 = [magnitudes(b1_idx) * 1, magnitudes(b2_idx) * 2];

    estimator = EGM(system, data, Knots=2:2:8, MaxIterationsGM=20, PCV = .1, ...
                    InitialConditions=(1+error)*[.6 .6 .1], ConvergenceTolGM=1e-4, Sigma=1E-2);
    try 
        out = estimator.estimate(beta0);
        dist_EGM = [{eucl_rel(system.k0, out.beta(:))} {out.beta}];

        converged_EGM(b1_idx, b2_idx, init_idx) = dist_EGM{1};
    catch ME
        disp(ME)
    end

    try 
        out2 = estimator.estimate_TM(beta0, (1+error)*[.6 .6 .1]);
        dist_TM = [{eucl_rel(system.k0, out2.beta(:))} {out2.beta}];

        converged_TM(b1_idx, b2_idx, init_idx) = dist_TM{1};
    catch ME
        disp(ME)
    end

    try 
        out3 = estimator.estimate_iterative(beta0, (1+error)*[.6 .6 .1]);
        dist_GM = [{eucl_rel(system.k0, out3.beta(:))} {out3.beta}];

        converged_GM(b1_idx, b2_idx, init_idx) = dist_GM{1};
    catch ME
        disp(ME)
    end
end
end
save(sprintf('simulation/convergence_GLV_L%d', length(observed)))
end
