%% Setup system ------------------------------------------------------------
clearvars
close all

% system = System('model_GLV.txt', 'auxiliary_GLV.txt', FixedParameters = ["r"]);
% save('system_GLV.mat', 'system')
load('system_GLV.mat')


%% Assess estimation accuracy ----------------------------------------------
observed = 2:system.K;                                                      % uncomment if L=2
% observed = system.K;                                                      % uncomment if L=1

dt_values = [.25 .5 1];
noise_levels = [.001 .02 .05 .1];
init_error = 5.^(-1:.5:1) - 1;
seeds = 1:50;

[betas_EGM, times_EGM, betas_GM, ...
 times_GM, betas_TM, times_TM] = deal(zeros(2, length(seeds), length(init_error), ...
                                      length(noise_levels), length(dt_values)));
[accuracies_EGM, ...
 accuracies_GM, accuracies_TM] = deal(zeros(length(seeds), length(init_error), ...
                                      length(noise_levels), length(dt_values)));
for dt_idx = 1:length(dt_values)
for noise_idx = 1:length(noise_levels)
for init_idx = 1:length(init_error)
for seed = 1:length(seeds)
    dt = dt_values(dt_idx);
    noise = noise_levels(noise_idx);
    error = init_error(init_idx);

    rng(10000*dt_idx + 1000*noise_idx + 100*init_idx + seed - 1)

    beta0 = .5*[1 2]; 

    generator = Generator(system, N=1, t=0:dt:10, error_std=noise, D_mult=eps, observed=observed);
    [data, ground_truth] = generator.generate();

    estimator = EGM(system, data, Knots=2:2:8, MaxIterationsGM=20, PCV = .1, ...
                    InitialConditions=(1+error)*[.6 .6 .1], ConvergenceTolGM=1e-4, Sigma=1E-2);
    try 
        out = estimator.estimate(beta0);
        dist_EGM = [{eucl_rel(system.k0, out.beta(:))} {out.beta}];

        betas_EGM(:, seed, init_idx, noise_idx, dt_idx) = out.beta;
        times_EGM(:, seed, init_idx, noise_idx, dt_idx) = out.time;
        accuracies_EGM(seed, init_idx, noise_idx, dt_idx) = eucl_rel(system.k0, out.beta(:));

    catch ME
        disp(ME)
        times_EGM(:, seed, init_idx, noise_idx, dt_idx) = [99 99];
        accuracies_EGM(seed, init_idx, noise_idx, dt_idx) = 99;
    end

    try 
        out2 = estimator.estimate_iterative(beta0, (1+error)*[.6 .6 .1]);
        dist_GM = [{eucl_rel(system.k0, out2.beta(:))} {out2.beta}];

        betas_GM(:, seed, init_idx, noise_idx, dt_idx) = out2.beta;
        times_GM(:, seed, init_idx, noise_idx, dt_idx) = out2.time;
        accuracies_GM(seed, init_idx, noise_idx, dt_idx) = eucl_rel(system.k0, out2.beta(:));

    catch ME
        disp(ME)
        times_GM(:, seed, init_idx, noise_idx, dt_idx) = [99 99];
        accuracies_GM(seed, init_idx, noise_idx, dt_idx) = 99;
    end

    try 
        out3 = estimator.estimate_TM(beta0, (1+error)*[.6 .6 .1]);
        dist_TM = [{eucl_rel(system.k0, out3.beta(:))} {out3.beta}];

        betas_TM(:, seed, init_idx, noise_idx, dt_idx) = out3.beta;
        times_TM(:, seed, init_idx, noise_idx, dt_idx) = out3.time;
        accuracies_TM(seed, init_idx, noise_idx, dt_idx) = eucl_rel(system.k0, out3.beta(:));

    catch ME
        disp(ME)
        times_TM(:, seed, init_idx, noise_idx, dt_idx) = [99 99];
        accuracies_TM(seed, init_idx, noise_idx, dt_idx) = 99;
    end
end
end
end
save(sprintf('simulation/convergence_GLV_L%d', length(observed)))
end
