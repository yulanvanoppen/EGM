%% Setup system ------------------------------------------------------------
clearvars
close all

% system = System('model_HS.txt', 'auxiliary_HS.txt');
% save('system_HS.mat', 'system')
load('system_HS.mat')


%% Generate data -----------------------------------------------------------
dt = .5;
noise = .01;
init_error = 5.^(-1:.5:1) - 1;

generator = Generator(system, N=1, t=0:dt:10, error_std=noise, ...
                      D_mult=eps, observed=2:system.K);
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
    beta0 = [magnitudes(b1_idx) * .95, magnitudes(b2_idx) * 1];

    estimator = EGM(system, data, Knots=[2.5 3.75 5 6.25 7.5], MaxIterationsGM=20, PCV = .25, ...
                    InitialConditions=(1+error)*[1 1], ConvergenceTolGM=1e-4, Sigma=1E-2);
    try 
        out = estimator.estimate(beta0);
        dist_EGM = [{eucl_rel(system.k0, out.beta(:))} {out.beta}];

        converged_EGM(b1_idx, b2_idx, init_idx) = dist_EGM{1};
    catch ME
        disp(ME)
    end

    try 
        out2 = estimator.estimate_TM(beta0, (1+error)*[1 1]);
        dist_TM = [{eucl_rel(system.k0, out2.beta(:))} {out2.beta}];
    
        converged_TM(b1_idx, b2_idx, init_idx) = dist_TM{1};

    catch ME
        disp(ME)
    end

    try 
        out3 = estimator.estimate_iterative(beta0, (1+error)*[1 1]);
        dist_GM = [{eucl_rel(system.k0, out3.beta(:))} {out3.beta}];

        converged_GM(b1_idx, b2_idx, init_idx) = dist_GM{1};
    catch ME
        disp(ME)
    end
end
end
save('simulation/convergence_HS')
end

